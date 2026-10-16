"""Stage 4: steered segmentation + semantics + move pairing.

Geometry (stage 3) already said WHERE something changed. This stage adds WHAT and pairs
removed <-> added clusters into moves. For each removed/added cluster:

1. Pick up to 2 frames of the owning session where the most voxels are visible
   (reuses show_clusters.best_frames with kind=1 = "match" votes).
2. Project the voxels into each frame and use their 2D bounding box as a SAM box prompt.
   SAM returns a tight mask of the object; crop that mask's bbox from the frame.
3. Embed each crop with DINOv2 and caption it with BLIP. Save crops to out/crops/.
4. Pair removed <-> added greedy best-first by DINOv2 cosine (threshold COS_MIN) with a
   size-ratio sanity check; record the 3D distance between centres as the move length.
   Pairs closer than MOVE_MIN get flagged "likely alignment noise" and demoted to low.
5. Score confidence per cluster/move from see-through votes, voxel count, how close the
   crop sits to the image edge, and whether a would-be move has the same caption on both
   ends (the c8/c22 "monitor" trap in CLAUDE.md).

Writes data/semantics.json and out/crops/c<id>_f<frame>.png.
Usage: python semantics.py
"""
import json, os
import numpy as np
import torch
from PIL import Image
from transformers import (SamModel, SamProcessor,
                          AutoModel, AutoImageProcessor,
                          BlipForConditionalGeneration, BlipProcessor)
from recon import load, project, good_pixels, to_room
from show_clusters import best_frames

N_FRAMES = 2
BOX_PAD = 8               # pixels of padding around the projected-voxel box fed to SAM
CROP_PAD = 6              # pixels of padding around the mask bbox when cropping
EDGE_MARGIN = 15          # pixels: crops closer than this to the frame edge are "edgy"
COS_MIN = 0.55            # DINOv2 cosine threshold for pairing
SIZE_RATIO_MAX = 2.5      # max(size_a, size_b) / min(size_a, size_b)
MOVE_MIN = 0.30           # metres: below this a "move" is just alignment noise


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def project_box(d, X, i):
    """2D bbox of projected voxels in frame i, clipped and padded. None if too few / too small."""
    u, v, z = project(d, X, i)
    ok = (z > 0.1) & (u >= 0) & (u < d["W"]) & (v >= 0) & (v < d["H"])
    if ok.sum() < 3:
        return None
    u, v = u[ok], v[ok]
    x0, y0 = max(int(u.min()) - BOX_PAD, 0), max(int(v.min()) - BOX_PAD, 0)
    x1, y1 = min(int(u.max()) + BOX_PAD, d["W"] - 1), min(int(v.max()) + BOX_PAD, d["H"] - 1)
    return [x0, y0, x1, y1] if (x1 - x0 >= 4 and y1 - y0 >= 4) else None


def sam_mask(sam, sam_proc, device, img, box):
    """SAM mask inside a box prompt. img = PIL.Image. Returns bool (H,W)."""
    inp = sam_proc(img, input_boxes=[[box]], return_tensors="pt")
    inp = {k: (v.to(torch.float32) if torch.is_tensor(v) and v.is_floating_point() else v).to(device)
           for k, v in inp.items()}
    with torch.no_grad():
        out = sam(**inp, multimask_output=True)
    masks = sam_proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0][0]
    return masks[int(np.argmax(out.iou_scores[0, 0].cpu().numpy()))].numpy().astype(bool)


def crop_from_mask(img_np, mask):
    """Crop img_np (H,W,3) to mask's bbox with padding. Return (PIL.Image, cbox) or (None, None)."""
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return None, None
    x0, x1 = max(xs.min() - CROP_PAD, 0), min(xs.max() + CROP_PAD, img_np.shape[1] - 1)
    y0, y1 = max(ys.min() - CROP_PAD, 0), min(ys.max() + CROP_PAD, img_np.shape[0] - 1)
    return Image.fromarray(img_np[y0:y1 + 1, x0:x1 + 1]), (x0, y0, x1, y1)


def dino_embed(dino, dino_proc, device, img):
    inp = dino_proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        v = dino(**inp).last_hidden_state[0, 0].cpu().numpy()        # CLS token
    return v / (np.linalg.norm(v) + 1e-9)


def caption(blip, blip_proc, device, img):
    inp = blip_proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = blip.generate(**inp, max_new_tokens=20)
    return blip_proc.decode(ids[0], skip_special_tokens=True).strip()


def edge_dist(cbox, W, H):
    return min(cbox[0], cbox[1], W - 1 - cbox[2], H - 1 - cbox[3])


def process_cluster(c, X, d, good, models):
    """Build a semantic record for one cluster: best frames, SAM crops, embedding, caption."""
    sam, sam_proc, dino, dino_proc, blip, blip_proc, device = models
    frames = np.where(d["video_id"] == c["present_in"])[0]
    rec = {"id": c["id"], "verdict": c["verdict"], "present_in": int(c["present_in"]),
           "frames": [], "crops": [], "captions": [], "edge_dists": []}
    embs = []
    for i, n_visible in best_frames(d, good, X, frames, kind=1, k=N_FRAMES):
        box = project_box(d, X, i)
        if box is None:
            continue
        try:
            mask = sam_mask(sam, sam_proc, device, Image.fromarray(d["images"][i]), box)
        except Exception as e:
            print(f"  c{c['id']} frame {int(d['frame_idx'][i])}: SAM failed ({e})")
            continue
        crop, cbox = crop_from_mask(d["images"][i], mask)
        if crop is None:
            continue
        name = f"c{c['id']}_f{int(d['frame_idx'][i])}.png"
        crop.save(f"out/crops/{name}")
        cap = caption(blip, blip_proc, device, crop)
        rec["frames"].append(int(d["frame_idx"][i]))
        rec["crops"].append(f"out/crops/{name}")
        rec["captions"].append(cap)
        rec["edge_dists"].append(edge_dist(cbox, d["W"], d["H"]))
        embs.append(dino_embed(dino, dino_proc, device, crop))
        print(f"  c{c['id']} s{c['present_in']} frame {int(d['frame_idx'][i])}: '{cap}' "
              f"(edge {rec['edge_dists'][-1]} px, {n_visible} vox visible)")
    if embs:
        v = np.mean(embs, 0); rec["embedding"] = (v / (np.linalg.norm(v) + 1e-9)).tolist()
    else:
        rec["embedding"] = None
    return rec


def bbox_size(c):
    return np.subtract(c["bbox_max"], c["bbox_min"])


def pair_moves(records, clusters_by_id):
    """Greedy best-first removed <-> added pairing by DINOv2 cosine + size ratio."""
    rem = [r for r in records if r["verdict"] == "removed" and r["embedding"] is not None]
    add = [r for r in records if r["verdict"] == "added" and r["embedding"] is not None]
    scored = []
    for r in rem:
        for a in add:
            s_r, s_a = (np.linalg.norm(bbox_size(clusters_by_id[k])) for k in (r["id"], a["id"]))
            ratio = max(s_r, s_a) / max(min(s_r, s_a), 1e-6)
            cos = float(np.dot(r["embedding"], a["embedding"]))
            if cos >= COS_MIN and ratio <= SIZE_RATIO_MAX:
                dist = float(np.linalg.norm(np.subtract(clusters_by_id[r["id"]]["centre"],
                                                        clusters_by_id[a["id"]]["centre"])))
                scored.append((cos, ratio, dist, r["id"], a["id"]))
    scored.sort(reverse=True)
    used_r, used_a, moves = set(), set(), []
    for cos, ratio, dist, rid, aid in scored:
        if rid in used_r or aid in used_a:
            continue
        used_r.add(rid); used_a.add(aid)
        moves.append(dict(removed=rid, added=aid, cos=round(cos, 3),
                          size_ratio=round(ratio, 2), distance_m=round(dist, 2)))
    return moves, used_r, used_a


def confidence(c, rec, paired=False, partner_caption=None):
    """Rule-based score. Returns ('high'/'medium'/'low', reasons)."""
    see, n = c["see_through"], c["n_voxels"]
    reasons, score = [], 0
    if see >= 10: score += 2; reasons.append(f"see-through {see}")
    elif see >= 5: score += 1
    else: score -= 1; reasons.append(f"only {see} see-through votes")
    if n >= 100: score += 1
    if rec["edge_dists"] and min(rec["edge_dists"]) >= EDGE_MARGIN: score += 1
    elif rec["edge_dists"]:
        score -= 1; reasons.append(f"crop {min(rec['edge_dists'])} px from frame edge")
    if partner_caption and rec["captions"] and rec["captions"][0].lower() == partner_caption.lower():
        reasons.append(f"same caption on both ends ('{rec['captions'][0]}') -- probable misalignment")
        score -= 2
    if paired: score += 1
    level = "high" if score >= 3 else "medium" if score >= 1 else "low"
    return level, reasons


if __name__ == "__main__":
    os.makedirs("out/crops", exist_ok=True)
    d = to_room(load())
    good = good_pixels(d)
    ch = json.load(open("data/changes.json"))
    vox = np.load("data/changes_voxels.npz")
    clusters_by_id = {c["id"]: c for c in ch["clusters"]}
    changed = [c for c in ch["clusters"] if c["verdict"] in ("removed", "added")]

    device = pick_device()
    print(f"device: {device}")
    print("loading SAM (facebook/sam-vit-base)...")
    sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()
    print("loading DINOv2 (facebook/dinov2-small)...")
    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    print("loading BLIP (Salesforce/blip-image-captioning-base)...")
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device).eval()
    models = (sam, sam_proc, dino, dino_proc, blip, blip_proc, device)

    print(f"\nprocessing {len(changed)} removed/added clusters...")
    records = [process_cluster(c, vox[f"c{c['id']}"], d, good, models) for c in changed]

    moves, used_r, used_a = pair_moves(records, clusters_by_id)
    print(f"\n{len(moves)} candidate move(s) found:")
    rec_by_id = {r["id"]: r for r in records}
    for m in moves:
        rr, ra = rec_by_id[m["removed"]], rec_by_id[m["added"]]
        tag = "MISALIGN?" if m["distance_m"] < MOVE_MIN else "MOVE"
        cap_r = rr["captions"][0] if rr["captions"] else "?"
        cap_a = ra["captions"][0] if ra["captions"] else "?"
        print(f"  {tag}: c{m['removed']} '{cap_r}' -> c{m['added']} '{cap_a}' | "
              f"{m['distance_m']} m, cos {m['cos']}, size-ratio {m['size_ratio']}")

    out = {"moves": [], "singletons": []}
    for m in moves:
        rr, ra = rec_by_id[m["removed"]], rec_by_id[m["added"]]
        partner_cap_r = ra["captions"][0] if ra["captions"] else None
        partner_cap_a = rr["captions"][0] if rr["captions"] else None
        lvl_r, rs_r = confidence(clusters_by_id[m["removed"]], rr, True, partner_cap_r)
        lvl_a, rs_a = confidence(clusters_by_id[m["added"]], ra, True, partner_cap_a)
        overall = sorted([lvl_r, lvl_a], key=lambda s: ["low", "medium", "high"].index(s))[0]
        if m["distance_m"] < MOVE_MIN:
            overall = "low"; rs_r.append(f"only {m['distance_m']} m apart -- likely alignment noise")
        out["moves"].append(dict(**m,
                                 caption_from=rr["captions"][0] if rr["captions"] else None,
                                 caption_to=ra["captions"][0] if ra["captions"] else None,
                                 confidence=overall, reasons_from=rs_r, reasons_to=rs_a))
    for r in records:
        if r["id"] in used_r or r["id"] in used_a:
            continue
        lvl, reasons = confidence(clusters_by_id[r["id"]], r)
        out["singletons"].append(dict(id=r["id"], verdict=r["verdict"], present_in=r["present_in"],
                                      caption=r["captions"][0] if r["captions"] else None,
                                      frames=r["frames"], crops=r["crops"],
                                      centre=clusters_by_id[r["id"]]["centre"],
                                      volume_l=clusters_by_id[r["id"]]["volume_l"],
                                      confidence=lvl, reasons=reasons))

    json.dump(out, open("data/semantics.json", "w"), indent=1)
    print(f"\nwrote data/semantics.json ({len(out['moves'])} moves, {len(out['singletons'])} singletons)")
    print("crops in out/crops/")
