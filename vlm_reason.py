"""Stage 4 (VLM): voxel-cluster semantics via Claude Haiku 4.5.

For each removed/added cluster from stage 3:

1. Pick the best frame of the owning session where the cluster is most visible
   (reuses show_clusters.best_frames with kind=1 = "match" votes).
2. Project cluster voxels into that frame -> 2D pixel points.
3. Use SAM 2.1 with those points as a POINT PROMPT (not a box) to get a precise mask
   of the object the geometry is pointing at. This "reverses the reconstruction":
   3D voxels -> 2D projection -> SAM mask -> object boundary the VLM can see as one thing.
4. Save a context crop around the mask with the mask outlined in red and the voxel
   projection points scattered as red dots.
5. Call Claude Haiku 4.5 once per cluster to describe what's inside the outline and
   classify whether this looks like a real change or a reconstruction/alignment artefact.
6. For each (removed, added) pair whose 3D centres are within D_MAX metres, call Claude
   again with both crops and ask whether they are the same object, two visually similar
   but distinct instances, or unrelated.
7. Greedy pair by VLM confidence * geometric evidence. Residuals stay as removed/added.
8. Confidence levels combine geometry (see-through votes), VLM confidence, and crop quality.

Nothing in the prompts assumes a classroom or any particular object vocabulary -- the VLM
reads the scene itself. The pipeline works for any room.

Writes:
  data/vlm.json          per-cluster descriptions + pair decisions + confidences
  out/vlm_crops/         per-cluster crops (annotated and raw)
  out/report.md          human-readable markdown report

Usage: ANTHROPIC_API_KEY=... python vlm_reason.py
"""
import base64, io, json, os, sys, time
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from transformers import SamModel, SamProcessor
import anthropic
from recon import load, project, good_pixels, to_room
from show_clusters import best_frames

MODEL = "claude-haiku-4-5-20251001"
D_MAX_PAIR = 6.0            # metres: ignore pair candidates farther apart than this
D_SCALE = 4.0               # metres: distance discount scale in pair scoring
CROP_CTX = 16               # extra pixels of context around the mask bbox when cropping
MAX_POINT_PROMPTS = 8       # how many of the projected voxel points to feed SAM
MOVE_MIN = 0.30             # metres: below this a "move" is really alignment noise
PAIR_MIN_CONF = 0.80        # VLM pair confidence floor to even consider


# -------------------------------------------------------------------- helpers
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_f32(x, device):
    return {k: (v.to(torch.float32) if torch.is_tensor(v) and v.is_floating_point() else v).to(device)
            for k, v in x.items()}


def encode_image(pil):
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def extract_json(text):
    """Best-effort pull of a JSON object from a VLM response."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}") + 1
        if s >= 0 and e > s:
            try:
                return json.loads(text[s:e])
            except Exception:
                pass
    return None


def sam_mask_from_points(sam, sam_proc, device, pil, points_2d):
    """SAM mask with multiple foreground point prompts. points_2d: (N, 2) pixel coords."""
    pts = points_2d.tolist()
    labels = [1] * len(pts)
    inp = sam_proc(pil, input_points=[[pts]], input_labels=[[labels]], return_tensors="pt")
    inp = to_f32(inp, device)
    with torch.no_grad():
        out = sam(**inp, multimask_output=True)
    masks = sam_proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0][0]
    scores = out.iou_scores[0, 0].cpu().numpy()
    return masks[int(np.argmax(scores))].numpy().astype(bool)


def crop_and_annotate(pil, mask, points_2d):
    """Return (clean_crop, annotated_crop, cbox).

    The CLEAN crop is a tight raw-pixel crop centred on the SAM mask (fed to the VLM so
    annotations don't contaminate its description). The ANNOTATED crop is the same region
    with a thin 1-pixel red outline of the mask (saved for human inspection only).
    """
    arr = np.array(pil.convert("RGB"))
    H, W = arr.shape[:2]
    ys, xs = np.where(mask)
    if len(xs) < 20:
        xs, ys = points_2d[:, 0].astype(int), points_2d[:, 1].astype(int)
    x0 = max(int(xs.min()) - CROP_CTX, 0); y0 = max(int(ys.min()) - CROP_CTX, 0)
    x1 = min(int(xs.max()) + CROP_CTX, W - 1); y1 = min(int(ys.max()) + CROP_CTX, H - 1)
    clean = Image.fromarray(arr[y0:y1 + 1, x0:x1 + 1])
    annotated_arr = arr.copy()
    edge = mask ^ ndimage.binary_erosion(mask)
    annotated_arr[edge] = [255, 40, 40]
    annotated_crop = Image.fromarray(annotated_arr[y0:y1 + 1, x0:x1 + 1])
    return clean, annotated_crop, (x0, y0, x1, y1)


def build_cluster_crop(d, good, X, c, sam, sam_proc, device, out_dir, rng):
    """One best frame of the owning session, SAM mask from projected voxels, annotated crop."""
    frames = np.where(d["video_id"] == c["present_in"])[0]
    picks = best_frames(d, good, X, frames, kind=1, k=1)
    if not picks:
        return None
    i, _ = picks[0]
    u, v, z = project(d, X, i)
    ok = (z > 0.1) & (u >= 0) & (u < d["W"]) & (v >= 0) & (v < d["H"])
    if ok.sum() < 3:
        return None
    pts = np.stack([u[ok], v[ok]], 1)
    # Spread the SAM prompt points across the projection so the mask covers the whole object
    if len(pts) > MAX_POINT_PROMPTS:
        idx = rng.choice(len(pts), MAX_POINT_PROMPTS, replace=False)
        prompt_pts = pts[idx]
    else:
        prompt_pts = pts
    pil = Image.fromarray(d["images"][i])
    try:
        mask = sam_mask_from_points(sam, sam_proc, device, pil, prompt_pts)
    except Exception as e:
        print(f"  c{c['id']} frame {int(d['frame_idx'][i])}: SAM failed ({e}); using point bbox only")
        mask = np.zeros((d["H"], d["W"]), bool)
        mask[v[ok].astype(int), u[ok].astype(int)] = True
    clean, annotated, cbox = crop_and_annotate(pil, mask, pts)
    frame_idx = int(d["frame_idx"][i])
    out_clean = f"{out_dir}/c{c['id']}_f{frame_idx}_crop.png"
    out_annot = f"{out_dir}/c{c['id']}_f{frame_idx}_annot.png"
    clean.save(out_clean); annotated.save(out_annot)
    return dict(frame_i=int(i), frame_idx=frame_idx, crop_path=out_clean,
                annot_path=out_annot, cbox=list(cbox), n_mask_px=int(mask.sum()),
                edge_px=min(cbox[0], cbox[1], d["W"] - 1 - cbox[2], d["H"] - 1 - cbox[3]))


# -------------------------------------------------------------------- VLM calls
SYS_DESCRIBE = (
    "You will see one tight crop from a photograph of a physical space. A 3D-geometry step "
    "flagged this region as a candidate change between two viewpoints of the same space -- "
    "the crop is already zoomed in to that region, so the main subject of the crop IS the "
    "thing in question.\n\n"
    "Describe the main subject of the crop in 1-2 short noun phrases "
    "(e.g. 'blue plastic cube on a desk', 'folded dark bag', 'window with a lowered beige blind'). "
    "Do NOT describe camera annotations or markers; describe real physical content only.\n\n"
    "Then classify the change as either a real change (an object was added, removed, or moved) "
    "or an alignment/depth artefact (the object is essentially still in the same place but the "
    "geometry shifted by a small amount) or uncertain. Keep reasoning short.\n\n"
    "Respond ONLY with valid JSON of this exact shape:\n"
    "{\n"
    "  \"description\": \"short noun phrase\",\n"
    "  \"significance\": \"real_change\" | \"alignment_artefact\" | \"uncertain\",\n"
    "  \"confidence\": 0.0-1.0,\n"
    "  \"why\": \"one short sentence\"\n"
    "}"
)

SYS_PAIR = (
    "You will see TWO tight crops from two photographs of the same physical space taken from "
    "different viewpoints. A 3D-geometry step flagged the region in each crop as a candidate "
    "change (the main subject of each crop is the thing in question).\n\n"
    "Decide whether the main subjects of the two crops are:\n"
    "  - 'same_object': you are confident these are the SAME specific physical instance.\n"
    "    Reserve this for cases where distinguishing features (colour, logo, exact shape, "
    "    orientation, markings) are visibly consistent across the two crops.\n"
    "  - 'similar_but_distinct': they look like the same kind of thing (same category or "
    "    visually similar) but you cannot verify they are the same specific instance.\n"
    "    Default to this when there is any ambiguity or when the match is only generic.\n"
    "  - 'unrelated': clearly different objects.\n\n"
    "Be conservative. If you cannot see a distinguishing feature that uniquely identifies "
    "the object across both crops, prefer 'similar_but_distinct' over 'same_object'.\n\n"
    "Respond ONLY with valid JSON of this exact shape:\n"
    "{\n"
    "  \"relation\": \"same_object\" | \"similar_but_distinct\" | \"unrelated\",\n"
    "  \"confidence\": 0.0-1.0,\n"
    "  \"why\": \"one short sentence citing the distinguishing (or missing) feature\"\n"
    "}"
)


def ask_describe(client, crop, verdict):
    msg = client.messages.create(
        model=MODEL, max_tokens=400,
        system=[{"type": "text", "text": SYS_DESCRIBE,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": encode_image(crop)}},
            {"type": "text", "text": f"The geometry step called this region: {verdict}."},
        ]}])
    text = msg.content[0].text
    parsed = extract_json(text) or {"description": text.strip()[:200],
                                    "significance": "uncertain", "confidence": 0.0, "why": ""}
    parsed["_tokens"] = {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens}
    return parsed


def ask_pair(client, crop_a, crop_b, dist_m):
    msg = client.messages.create(
        model=MODEL, max_tokens=300,
        system=[{"type": "text", "text": SYS_PAIR,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": encode_image(crop_a)}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": encode_image(crop_b)}},
            {"type": "text", "text": f"The two outlined regions are {dist_m:.2f} metres apart in 3D."},
        ]}])
    text = msg.content[0].text
    parsed = extract_json(text) or {"relation": "unrelated", "confidence": 0.0, "why": text.strip()[:200]}
    parsed["_tokens"] = {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens}
    return parsed


# -------------------------------------------------------------------- confidence
def confidence_level(cluster, crop_meta, vlm_desc, vlm_pair=None, dist_m=None, noise_floor=0.16):
    """Combine geometry, VLM reasoning, and crop quality into a bucket."""
    see, iv = cluster["see_through"], max(cluster["in_view"], 1)
    reasons, score = [], 0
    frac_empty = see / iv
    if frac_empty >= 0.8 and see >= 5: score += 2; reasons.append(f"{see}/{iv} empty-votes")
    elif frac_empty >= 0.5 and see >= 3: score += 1; reasons.append(f"{see}/{iv} empty-votes")
    else: score -= 1; reasons.append(f"only {see}/{iv} empty-votes")
    if cluster["n_voxels"] >= 100: score += 1
    if crop_meta["edge_px"] >= 15: score += 1
    else: score -= 1; reasons.append(f"crop {crop_meta['edge_px']} px from frame edge")
    if vlm_desc.get("significance") == "alignment_artefact":
        score -= 2; reasons.append("VLM: alignment artefact (" + vlm_desc.get("why", "")[:60] + ")")
    elif vlm_desc.get("significance") == "real_change":
        score += 1; reasons.append("VLM: real change")
    if vlm_pair:
        if vlm_pair["relation"] == "same_object" and vlm_pair["confidence"] >= 0.7:
            score += 2; reasons.append(f"VLM pair: same object (conf {vlm_pair['confidence']:.2f})")
        elif vlm_pair["relation"] == "similar_but_distinct":
            reasons.append("VLM pair: similar but distinct instances")
        if dist_m is not None and dist_m < 2 * noise_floor:
            score -= 1; reasons.append(f"move {dist_m:.2f} m near noise floor")
    level = "high" if score >= 3 else "medium" if score >= 1 else "low"
    return level, reasons


# -------------------------------------------------------------------- main
if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set in environment. Export it and re-run.")
    os.makedirs("out/vlm_crops", exist_ok=True)
    rng = np.random.default_rng(0)

    print("loading reconstruction + stage 3 output...", flush=True)
    d = to_room(load())
    good = good_pixels(d)
    ch = json.load(open("data/changes.json"))
    vox = np.load("data/changes_voxels.npz")
    clusters = [c for c in ch["clusters"] if c["verdict"] in ("removed", "added")]
    print(f"  {len(clusters)} removed/added clusters to reason about", flush=True)

    device = pick_device()
    print(f"loading SAM on {device}...", flush=True)
    sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()

    # ---------- Step 1: build an annotated crop per cluster ----------
    print("\nbuilding annotated crops (SAM point-prompt masks)...", flush=True)
    crops = {}
    for c in clusters:
        X = vox[f"c{c['id']}"]
        meta = build_cluster_crop(d, good, X, c, sam, sam_proc, device, "out/vlm_crops", rng)
        if meta is None:
            print(f"  c{c['id']}: no visible frame; skipping")
            continue
        crops[c["id"]] = meta
        print(f"  c{c['id']} ({c['verdict']:>7s})  frame {meta['frame_idx']}  "
              f"mask {meta['n_mask_px']} px  edge {meta['edge_px']} px", flush=True)

    # ---------- Step 2: VLM describe + significance per cluster ----------
    print("\ncalling Claude Haiku to describe + classify each cluster...", flush=True)
    client = anthropic.Anthropic()
    descriptions = {}
    tok_in = tok_out = 0
    for c in clusters:
        if c["id"] not in crops:
            continue
        crop = Image.open(crops[c["id"]]["crop_path"]).convert("RGB")
        try:
            vd = ask_describe(client, crop, c["verdict"])
        except Exception as e:
            print(f"  c{c['id']}: API error ({e}); skipping")
            continue
        descriptions[c["id"]] = vd
        tok_in += vd["_tokens"]["in"]; tok_out += vd["_tokens"]["out"]
        print(f"  c{c['id']} ({c['verdict']:>7s}): '{vd.get('description', '?')}'  "
              f"[{vd.get('significance', '?')} {vd.get('confidence', 0):.2f}]  "
              f"{vd.get('why','')[:70]}", flush=True)
        time.sleep(0.2)

    # ---------- Step 3: VLM pair asks ----------
    # Only ask about pairs where BOTH sides were flagged real_change by the describe step.
    # Dropping 'alignment_artefact' is obvious; also drop 'uncertain' because those crops
    # were too small/blurry for the VLM to describe reliably -- it will invent matches.
    def eligible(cid):
        return descriptions.get(cid, {}).get("significance") == "real_change"
    print("\ncalling Claude Haiku on candidate pairs...", flush=True)
    removed = [c for c in clusters if c["verdict"] == "removed" and c["id"] in crops and eligible(c["id"])]
    added   = [c for c in clusters if c["verdict"] == "added"   and c["id"] in crops and eligible(c["id"])]
    print(f"  eligible removed: {[c['id'] for c in removed]}", flush=True)
    print(f"  eligible added:   {[c['id'] for c in added]}", flush=True)
    pair_results = []
    for a in removed:
        for b in added:
            dist = float(np.linalg.norm(np.subtract(a["centre"], b["centre"])))
            if dist > D_MAX_PAIR:
                continue
            crop_a = Image.open(crops[a["id"]]["crop_path"]).convert("RGB")
            crop_b = Image.open(crops[b["id"]]["crop_path"]).convert("RGB")
            try:
                vp = ask_pair(client, crop_a, crop_b, dist)
            except Exception as e:
                print(f"  c{a['id']}<->c{b['id']}: API error ({e}); skipping")
                continue
            tok_in += vp["_tokens"]["in"]; tok_out += vp["_tokens"]["out"]
            pair_results.append(dict(a=a["id"], b=b["id"], dist_m=round(dist, 3),
                                     relation=vp.get("relation"), confidence=vp.get("confidence", 0),
                                     why=vp.get("why", "")))
            print(f"  c{a['id']:>2d} <-> c{b['id']:<2d}  {dist:.2f} m  "
                  f"{vp.get('relation','?'):>22s}  conf {vp.get('confidence',0):.2f}  "
                  f"{vp.get('why','')[:60]}", flush=True)
            time.sleep(0.2)

    # ---------- Step 4: greedy pair assignment with distance discount ----------
    # Score = VLM confidence * exp(-dist / D_SCALE). Close same_object matches dominate.
    scored, uncertain_pairs = [], []
    for p in pair_results:
        if p["confidence"] < PAIR_MIN_CONF and p["relation"] != "similar_but_distinct":
            continue
        s = p["confidence"] * float(np.exp(-p["dist_m"] / D_SCALE))
        if p["relation"] == "same_object":
            scored.append((s, p))
        elif p["relation"] == "similar_but_distinct" and p["dist_m"] < 1.5:
            # close "same kind, cannot verify" pairs are useful to surface as uncertain
            p["score"] = round(s, 3); uncertain_pairs.append(p)
    scored.sort(key=lambda x: -x[0])
    used_a, used_b, moves = set(), set(), []
    for s, p in scored:
        if p["a"] in used_a or p["b"] in used_b:
            continue
        used_a.add(p["a"]); used_b.add(p["b"])
        p["score"] = round(s, 3)
        moves.append(p)
    # keep only uncertain pairs whose IDs weren't absorbed by a confirmed move
    uncertain_pairs = [p for p in uncertain_pairs if p["a"] not in used_a and p["b"] not in used_b]
    print(f"\n{len(moves)} confirmed pair(s), {len(uncertain_pairs)} close 'similar_but_distinct' pair(s)")
    for m in moves:
        tag = "MISALIGN?" if m["dist_m"] < MOVE_MIN else "MOVE"
        print(f"  {tag}: c{m['a']} <-> c{m['b']}  {m['dist_m']:.2f} m  "
              f"score {m['score']:.2f}  ({m['why'][:70]})", flush=True)
    for p in uncertain_pairs:
        print(f"  UNCERTAIN: c{p['a']} <-> c{p['b']}  {p['dist_m']:.2f} m  "
              f"({p['why'][:70]})", flush=True)

    # ---------- Step 5: finalise output ----------
    cluster_by_id = {c["id"]: c for c in clusters}

    def with_desc(cid, extra=None, pair=None, dist=None):
        c = cluster_by_id[cid]
        lvl, reasons = confidence_level(c, crops[cid], descriptions.get(cid, {}),
                                        vlm_pair=pair, dist_m=dist)
        rec = dict(id=cid, verdict=c["verdict"], centre=c["centre"], bbox_min=c["bbox_min"],
                   bbox_max=c["bbox_max"], volume_l=c["volume_l"],
                   see_through=c["see_through"], in_view=c["in_view"],
                   frame=crops[cid]["frame_idx"], crop=crops[cid]["crop_path"],
                   annot=crops[cid]["annot_path"],
                   description=descriptions.get(cid, {}).get("description"),
                   significance=descriptions.get(cid, {}).get("significance"),
                   vlm_confidence=descriptions.get(cid, {}).get("confidence"),
                   vlm_why=descriptions.get(cid, {}).get("why"),
                   confidence=lvl, reasons=reasons)
        if extra: rec.update(extra)
        return rec

    out = {"moves": [], "misalignments": [], "uncertain_pairs": [],
           "singletons": [], "artefacts": []}
    for p in uncertain_pairs:
        a_rec = with_desc(p["a"], pair=p, dist=p["dist_m"])
        b_rec = with_desc(p["b"], pair=p, dist=p["dist_m"])
        out["uncertain_pairs"].append(dict(from_=a_rec, to=b_rec, distance_m=p["dist_m"],
                                           pair_relation=p["relation"],
                                           pair_confidence=p["confidence"], pair_why=p["why"],
                                           confidence="low"))
        used_a.add(p["a"]); used_b.add(p["b"])    # don't also report them as singletons
    for m in moves:
        a_rec = with_desc(m["a"], pair=m, dist=m["dist_m"])
        b_rec = with_desc(m["b"], pair=m, dist=m["dist_m"])
        # overall confidence = min of the two sides
        overall = sorted([a_rec["confidence"], b_rec["confidence"]],
                         key=lambda x: ["low", "medium", "high"].index(x))[0]
        rec = dict(from_=a_rec, to=b_rec, distance_m=m["dist_m"],
                   pair_relation=m["relation"], pair_confidence=m["confidence"],
                   pair_why=m["why"], pair_score=m.get("score"), confidence=overall)
        if m["dist_m"] < MOVE_MIN:
            rec["confidence"] = "low"
            rec["note"] = f"only {m['dist_m']:.2f} m apart -- likely alignment noise"
            out["misalignments"].append(rec)
        else:
            out["moves"].append(rec)

    for c in clusters:
        if c["id"] not in crops or c["id"] in used_a or c["id"] in used_b:
            continue
        rec = with_desc(c["id"])
        if rec["significance"] == "alignment_artefact":
            out["artefacts"].append(rec)
        else:
            out["singletons"].append(rec)

    json.dump(out, open("data/vlm.json", "w"), indent=1)
    print(f"\nwrote data/vlm.json  ({len(out['moves'])} moves, "
          f"{len(out['singletons'])} singletons, {len(out['artefacts'])} artefacts flagged by VLM)")
    print(f"API usage: {tok_in} input tokens, {tok_out} output tokens")

    # ---------- Step 6: markdown report ----------
    BADGE = {"high": "**[HIGH]**", "medium": "**[MED]**", "low": "**[LOW]**"}
    lines = ["# What changed in the room — report (geometry + VLM reasoning)\n"]
    lines.append("Pipeline: Depth Anything 3 shared map -> SuperMap voxel visibility diff "
                 "(stage 3) -> SAM point-prompt masks of each flagged region -> Claude Haiku 4.5 "
                 "describes the object and judges whether each change is meaningful; "
                 "same-object pair decisions on candidate moves.\n")
    lines.append(f"Totals: **{len(out['moves'])} move(s), {len(out['singletons'])} singleton(s), "
                 f"{len(out['artefacts'])} suppressed as alignment artefact(s)**.\n")

    lines.append("## Findings")
    for i, m in enumerate(out["moves"], 1):
        f, t = m["from_"], m["to"]
        f_desc, t_desc = f["description"] or "?", t["description"] or "?"
        label = f_desc if f_desc.lower() == t_desc.lower() else f"{f_desc}  ⇄  {t_desc}"
        lines.append(f"\n### M{i} {BADGE.get(m['confidence'], BADGE['medium'])} Moved: "
                     f"{label}  —  {m['distance_m']} m")
        lines.append(f"- From {tuple(round(x,2) for x in f['centre'])} to "
                     f"{tuple(round(x,2) for x in t['centre'])}")
        lines.append(f"- VLM: same object (conf {m['pair_confidence']:.2f}) — {m['pair_why']}")
        lines.append(f"- From-crop: `{f['crop']}` · To-crop: `{t['crop']}`")
        lines.append(f"- Geometry: {f['see_through']}/{f['in_view']} empty-votes (from), "
                     f"{t['see_through']}/{t['in_view']} empty-votes (to)")

    if out["misalignments"]:
        lines.append("\n## Likely alignment / no-move pairs (same object, below MOVE_MIN)")
        for i, m in enumerate(out["misalignments"], 1):
            f, t = m["from_"], m["to"]
            lines.append(f"\n### A{i} {BADGE['low']} {f['description']} <-> {t['description']}  —  {m['distance_m']} m")
            lines.append(f"- VLM pair: same object (conf {m['pair_confidence']:.2f}) — {m['pair_why']}")
            lines.append(f"- Note: {m.get('note','')}")
            lines.append(f"- Crops: `{f['crop']}` · `{t['crop']}`")

    if out["uncertain_pairs"]:
        lines.append("\n## Possible same object (uncertain, VLM called similar_but_distinct)")
        for i, p in enumerate(out["uncertain_pairs"], 1):
            f, t = p["from_"], p["to"]
            lines.append(f"\n### U{i} {BADGE['low']} {f['description']} <-> {t['description']}  —  {p['distance_m']} m")
            lines.append(f"- VLM pair: similar_but_distinct (conf {p['pair_confidence']:.2f}) — {p['pair_why']}")
            lines.append(f"- Crops: `{f['crop']}` · `{t['crop']}`")

    rank = {"high": 0, "medium": 1, "low": 2}
    for lab, group in (("Removed/Added", out["singletons"]), ("Suppressed artefacts", out["artefacts"])):
        if not group:
            continue
        lines.append(f"\n## {lab}")
        for i, r in enumerate(sorted(group, key=lambda r: rank.get(r["confidence"], 1)), 1):
            lines.append(f"\n### {r['verdict'].upper()}  {BADGE.get(r['confidence'], BADGE['medium'])}  "
                         f"{r['description']}")
            lines.append(f"- At {tuple(round(x,2) for x in r['centre'])}, volume {r['volume_l']} L")
            lines.append(f"- Crop: `{r['crop']}`")
            lines.append(f"- Geometry: {r['see_through']}/{r['in_view']} empty-votes")
            lines.append(f"- VLM: {r['significance']} (conf {r['vlm_confidence']:.2f}) — {r['vlm_why']}")
            if r["reasons"]:
                lines.append(f"- Notes: {', '.join(r['reasons'])}")

    with open("out/report.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote out/report.md ({len(lines)} lines)")
