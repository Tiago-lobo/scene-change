"""Generate a sequence of visual slides for a video walkthrough of the pipeline.

Reads the artefacts already produced by the pipeline and renders self-contained PNG
slides into showcase/. The slides are designed to be fed to a video-making tool
(e.g. a browser-based LLM) which can narrate them in order.

Slides:
  showcase/01_input_frames.png       sample frames from each session
  showcase/02_shared_map.png         pooled 3D point cloud (top-down + side view)
  showcase/03_evidence_sheet.png     stage-3 cluster evidence sheet (copied from out/)
  showcase/04_steered_crops.png      SAM-masked crops derived from the 3D voxel points
  showcase/05_vlm_descriptions.png   each crop with the Haiku description + verdict
  showcase/06_pair_decisions.png     the two moved pairs side-by-side with pair reasoning
  showcase/07_findings_topdown.png   top-down map of the room with changes overlaid
"""
import json, os, shutil, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from recon import load, unproject, good_pixels, to_room, camera_centres

OUT = "showcase"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})


def savefig(fig, name):
    path = f"{OUT}/{name}"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


# -----------------------------------------------------------------------------
def slide1(d):
    """Sample frames from each session."""
    v0_frames = [i for i in np.where(d["video_id"] == 0)[0]][::6][:4]
    v1_frames = [i for i in np.where(d["video_id"] == 1)[0]][::6][:4]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, i in zip(axes[0], v0_frames):
        ax.imshow(d["images"][i]); ax.axis("off")
        ax.set_title(f"session A, frame {int(d['frame_idx'][i])}")
    for ax, i in zip(axes[1], v1_frames):
        ax.imshow(d["images"][i]); ax.axis("off")
        ax.set_title(f"session B, frame {int(d['frame_idx'][i])}")
    fig.suptitle("Input: two phone videos of the same room, taken at different times",
                 y=0.995)
    savefig(fig, "01_input_frames.png")


def slide2(d, good):
    """Top-down and side view of the pooled 3D point cloud."""
    pts = np.concatenate([unproject(d, i)[good[i]] for i in range(d["S"])])
    cols = np.concatenate([d["images"][i][good[i]] for i in range(d["S"])])
    _, first = np.unique(np.floor(pts / 0.04).astype(np.int64), axis=0, return_index=True)
    first = np.random.default_rng(0).permutation(first)[:150000]
    pts, cols = pts[first], cols[first]
    pts = pts[pts[:, 2] < 2.6]
    cols = cols[: len(pts)]
    rgb = cols.astype(np.float32) / 255

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(pts[:, 0], pts[:, 1], s=0.6, c=rgb, alpha=0.9)
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
    axes[0].set_aspect("equal")
    axes[0].set_title("top-down (x, y)")
    axes[0].invert_yaxis()
    cams = camera_centres(d)
    for s, col in ((0, "#1f77b4"), (1, "#9467bd")):
        cc = cams[d["video_id"] == s]
        axes[0].scatter(cc[:, 0], cc[:, 1], s=25, c=col, marker="D",
                        edgecolors="black", linewidths=0.6,
                        label=f"camera, session {chr(ord('A') + s)}")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].scatter(pts[:, 0], pts[:, 2], s=0.6, c=rgb, alpha=0.9)
    axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("z (m, up)")
    axes[1].set_aspect("equal")
    axes[1].set_title("side (x, z)")

    fig.suptitle("Shared 3D reconstruction: both sessions pooled in one metric frame "
                 "(floor at z = 0, walls along x / y)", y=0.995)
    savefig(fig, "02_shared_map.png")


def slide3():
    """Copy stage-3 evidence sheet verbatim."""
    src = "out/clusters_1.png"
    if os.path.exists(src):
        shutil.copyfile(src, f"{OUT}/03_evidence_sheet.png")
        print("wrote", f"{OUT}/03_evidence_sheet.png (copied from {src})")


def slide4(vlm, cluster_meta):
    """Grid of annotated crops with SAM mask outlined in red."""
    items = []
    for m in vlm["moves"]:
        items.append((m["from_"], "MOVE from"))
        items.append((m["to"], "MOVE to"))
    for m in vlm.get("misalignments", []):
        items.append((m["from_"], "MISALIGN"))
        items.append((m["to"], "MISALIGN"))
    for s in vlm["singletons"]:
        items.append((s, s["verdict"].upper()))
    for a in vlm.get("artefacts", [])[:3]:
        items.append((a, "artefact"))

    n = len(items)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.6))
    axes = np.atleast_2d(axes).flatten()
    for a in axes:
        a.axis("off")
    for ax, (item, tag) in zip(axes, items):
        path = item["annot"] if "annot" in item else item.get("crop")
        if path and os.path.exists(path):
            ax.imshow(Image.open(path))
        ax.set_title(f"c{item['id']} — {tag}", fontsize=10)
    fig.suptitle("SAM point-prompt masks derived from each stage-3 voxel cluster "
                 "(red outline = segmented object the VLM will reason about)",
                 y=0.995)
    savefig(fig, "04_steered_crops.png")


def wrap_text(s, width=50):
    return "\n".join(textwrap.wrap(s or "", width=width))


def slide5(vlm):
    """Each clean crop with the Haiku description + verdict next to it."""
    items = []
    for m in vlm["moves"]:
        items.append(m["from_"]); items.append(m["to"])
    for m in vlm.get("misalignments", []):
        items.append(m["from_"]); items.append(m["to"])
    for s in vlm["singletons"]:
        items.append(s)

    n = len(items)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.6))
    axes = np.atleast_2d(axes).flatten()
    for a in axes:
        a.axis("off")
    for ax, it in zip(axes, items):
        path = it.get("crop")
        if path and os.path.exists(path):
            ax.imshow(Image.open(path))
        desc = wrap_text(f"\"{it.get('description','?')}\"", 42)
        sig = it.get("significance", "?")
        colour = {"real_change": "#2ca02c", "alignment_artefact": "#9467bd",
                  "uncertain": "#7f7f7f"}.get(sig, "black")
        ax.set_title(f"c{it['id']}  ({sig} {it.get('vlm_confidence', 0):.2f})\n{desc}",
                     fontsize=9, color=colour)
    fig.suptitle("Claude Haiku 4.5 — describe + classify each flagged region "
                 "(green = real change, purple = alignment artefact, grey = uncertain)",
                 y=0.995)
    savefig(fig, "05_vlm_descriptions.png")


def slide6(vlm):
    """Side-by-side pair comparisons with VLM reasoning."""
    pairs = []
    for m in vlm["moves"]:
        pairs.append(("MOVE", m))
    for m in vlm.get("misalignments", []):
        pairs.append(("MISALIGNMENT", m))

    n = len(pairs)
    if n == 0:
        return
    # Each pair gets 2 rows: top = images (2 cols), bottom = reasoning text spanning both.
    fig = plt.figure(figsize=(13, 5.8 * n))
    gs = fig.add_gridspec(n * 2, 2, height_ratios=[3, 1] * n, hspace=0.35, wspace=0.1)
    for row, (tag, m) in enumerate(pairs):
        f, t = m["from_"], m["to"]
        ax_from = fig.add_subplot(gs[row * 2, 0])
        ax_to   = fig.add_subplot(gs[row * 2, 1])
        ax_txt  = fig.add_subplot(gs[row * 2 + 1, :])
        for ax, item, side in ((ax_from, f, "FROM"), (ax_to, t, "TO")):
            ax.axis("off")
            path = item.get("crop")
            if path and os.path.exists(path):
                ax.imshow(Image.open(path))
            ax.set_title(f"c{item['id']}  {side}  at {tuple(round(x,2) for x in item['centre'])}\n"
                         f"\"{item.get('description','?')}\"", fontsize=10)
        ax_txt.axis("off")
        text = (f"{tag} — {m['distance_m']:.2f} m    |    VLM pair ({m['pair_relation']}, "
                f"conf {m['pair_confidence']:.2f}):\n\"{m['pair_why']}\"")
        ax_txt.text(0.5, 0.5, wrap_text(text, 100),
                    ha="center", va="center", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.6",
                              facecolor="#f0f0f0", edgecolor="#999"),
                    transform=ax_txt.transAxes)
    fig.suptitle("Claude Haiku 4.5 — same_object / similar_but_distinct / unrelated for each "
                 "candidate pair; confirmed moves + the one close misalignment",
                 y=0.995)
    savefig(fig, "06_pair_decisions.png")


def slide7(d, good, vlm):
    """Top-down map of the room with all findings overlaid."""
    pts = np.concatenate([unproject(d, i)[good[i]] for i in range(d["S"])])
    pts = pts[pts[:, 2] < 2.6]
    _, first = np.unique(np.floor(pts[:, :2] / 0.05).astype(np.int64), axis=0, return_index=True)
    pts = pts[first]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.6, c="#dddddd", alpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2)

    for m in vlm["moves"]:
        f, t = m["from_"], m["to"]
        A, B = np.array(f["centre"][:2]), np.array(t["centre"][:2])
        ax.annotate("", xy=B, xytext=A,
                    arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=3,
                                    shrinkA=8, shrinkB=8))
        # label the arrow on the LESS cluttered side (below the midpoint)
        mid = (A + B) / 2
        offset_y = 0.35
        ax.annotate(f"MOVE {m['distance_m']:.2f} m",
                    xy=mid, xytext=(mid[0], mid[1] + offset_y),
                    fontsize=10, color="#b35800", fontweight="bold",
                    ha="center",
                    arrowprops=dict(arrowstyle="-", color="#b35800", alpha=0.3))
        ax.scatter(*A, s=90, c="#d62728", marker="o", edgecolors="black", zorder=5)
        ax.scatter(*B, s=90, c="#2ca02c", marker="o", edgecolors="black", zorder=5)
        ax.text(A[0], A[1] - 0.22, f"c{f['id']} FROM\n{f['description'][:28]}",
                fontsize=8, color="#8b0000", ha="center", va="bottom")
        ax.text(B[0], B[1] - 0.22, f"c{t['id']} TO\n{t['description'][:28]}",
                fontsize=8, color="#005500", ha="center", va="bottom")

    for m in vlm.get("misalignments", []):
        f, t = m["from_"], m["to"]
        A, B = np.array(f["centre"][:2]), np.array(t["centre"][:2])
        ax.plot([A[0], B[0]], [A[1], B[1]], "--", color="#bcbd22", lw=2)
        ax.scatter(*A, s=70, c="#bcbd22", marker="s", edgecolors="black", zorder=5)
        ax.text(A[0] + 0.25, A[1] + 0.08,
                f"MISALIGN {m['distance_m']:.2f} m\n{f['description'][:30]}",
                fontsize=8, color="#8a890f", ha="left", va="top")

    for s in vlm["singletons"]:
        pos = np.array(s["centre"][:2])
        col = {"removed": "#d62728", "added": "#2ca02c"}.get(s["verdict"], "grey")
        ax.scatter(*pos, s=70, c=col, marker="o", edgecolors="black", zorder=5)
        ax.text(pos[0] + 0.15, pos[1] + 0.15,
                f"c{s['id']} {s['verdict']}\n{s['description'][:28]}",
                fontsize=8, color=col, ha="left", va="bottom")

    cams = camera_centres(d)
    for s, col in ((0, "#1f77b4"), (1, "#9467bd")):
        cc = cams[d["video_id"] == s]
        ax.scatter(cc[:, 0], cc[:, 1], s=45, c=col, marker="D",
                   edgecolors="black", linewidths=0.6,
                   label=f"camera {chr(ord('A') + s)}")

    handles = [
        mpatches.Patch(color="#ff7f0e", label="MOVE arrow"),
        mpatches.Patch(color="#bcbd22", label="MISALIGN (same object, below 30 cm)"),
        mpatches.Patch(color="#d62728", label="removed only"),
        mpatches.Patch(color="#2ca02c", label="added only"),
    ]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0],
              loc="upper right", fontsize=9)
    ax.set_title("Final findings on top of the room (top-down)")
    savefig(fig, "07_findings_topdown.png")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("loading recon + VLM output...", flush=True)
    d = to_room(load())
    good = good_pixels(d)
    vlm = json.load(open("data/vlm.json"))
    cluster_meta = json.load(open("data/changes.json"))

    slide1(d)
    slide2(d, good)
    slide3()
    slide4(vlm, cluster_meta)
    slide5(vlm)
    slide6(vlm)
    slide7(d, good, vlm)
    print(f"\nall slides in {OUT}/")
