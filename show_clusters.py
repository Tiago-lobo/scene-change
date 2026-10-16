"""Picture sheet per change cluster: where it is in the frames of the session that has it, and the
other session's frames that saw through that spot. Cluster voxels are drawn as dots.

Usage: python show_clusters.py            -> out/clusters_*.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from recon import load, project, good_pixels, to_room
from diff import votes

PER_SHEET = 8


def best_frames(d, good, X, frames, kind, k=2):
    """Frames where the most cluster points are visible (kind=1, match) or seen through (kind=0)."""
    v = votes(d, good, X, frames)[kind].sum(1)
    order = np.argsort(-v)[:k]
    return [(frames[o], int(v[o])) for o in order if v[o] > 0]


def draw(ax, d, i, X, title, colour):
    ax.imshow(d["images"][i])
    u, v, z = project(d, X, i)
    ok = (z > 0) & (u >= 0) & (u < d["W"]) & (v >= 0) & (v < d["H"])
    ax.scatter(u[ok], v[ok], s=2, c=colour, alpha=0.5)
    ax.set_title(title, fontsize=8)


if __name__ == "__main__":
    d = to_room(load())
    good = good_pixels(d)
    ch = json.load(open("data/changes.json"))
    vox = np.load("data/changes_voxels.npz")
    show = [c for c in ch["clusters"] if c["verdict"] in ("removed", "added")]
    os.makedirs("out", exist_ok=True)
    for s0 in range(0, len(show), PER_SHEET):
        part = show[s0:s0 + PER_SHEET]
        fig, axes = plt.subplots(len(part), 4, figsize=(16, 2.4 * len(part)))
        for row, c in zip(np.atleast_2d(axes), part):
            for a in row:
                a.axis("off")
            X = vox[f"c{c['id']}"]
            here = best_frames(d, good, X, np.where(d["video_id"] == c["present_in"])[0], kind=1)
            there = best_frames(d, good, X, np.where(d["video_id"] == c["checked_by"])[0], kind=0)
            colour = "red" if c["verdict"] == "removed" else "lime"
            for a, (i, n) in zip(row[:2], here):
                draw(a, d, i, X, f"c{c['id']} {c['verdict']} | s{c['present_in']} frame {d['frame_idx'][i]}: {n} pts visible", colour)
            for a, (i, n) in zip(row[2:], there):
                draw(a, d, i, X, f"s{c['checked_by']} frame {d['frame_idx'][i]}: {n} pts seen through", "yellow")
            row[0].text(-10, 140, f"c{c['id']}\n{c['volume_l']} L\n({c['centre'][0]:.1f}, {c['centre'][1]:.1f}, {c['centre'][2]:.1f})",
                        ha="right", va="center", fontsize=9)
        plt.tight_layout()
        path = f"out/clusters_{s0 // PER_SHEET + 1}.png"
        plt.savefig(path, dpi=70)
        print(path)
