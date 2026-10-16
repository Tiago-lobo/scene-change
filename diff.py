"""Stage 3: visibility-aware 3D diff between the two sessions -> change clusters.

1. Every good pixel becomes a 3D point in the room frame; each session fills its own voxel grid.
   A voxel is occupied only if >= MIN_SUPPORT points land in it (present/absent, not counts).
2. Candidates: occupied in one session with nothing within one voxel in the other (slack for alignment).
3. Visibility test (after SuperMap): project each candidate into every frame of the OTHER session and
   compare its depth z with the depth d that frame observed at that pixel:
       d > z + tau     -> see-through: the camera saw past the spot, so it is empty there  (for change)
       |d - z| <= tau  -> match: something is there                                         (against)
       d < z - tau     -> occluded: something in front, no evidence
       out of view / filtered pixel -> no vote
   tau = 0.05 * z + VOXEL. Confirmed if see-through >= MIN_SEE_THROUGH and see-through > match.
   No usable vote = not compared: "unseen" if no frame of the other session even looked there,
   "hidden" if it looked but something was always in front.
4. Voxels of the same verdict within MERGE_GAP of each other form one cluster; small ones are dropped.

Output: data/changes.json (clusters) and data/changes_voxels.npz (voxel centres per cluster).
Usage: python diff.py
"""
import json
import numpy as np
from scipy import ndimage
from recon import load, unproject, project, good_pixels, to_room

VOXEL = 0.03
MIN_SUPPORT = 5
MIN_SEE_THROUGH = 3
TAU_REL = 0.05
MERGE_GAP = 2             # voxels: pieces closer than this join one cluster
MIN_CLUSTER = 15          # voxels (~0.4 litres)
MIN_NOT_COMPARED = 300    # only report big not-compared regions
FLOOR_MARGIN = 0.04       # ignore voxels this close to the floor (floor depth noise)


def votes(d, good, X, frames, tau_rel=TAU_REL, voxel=VOXEL):
    """(n_frames, N) arrays: see-through, match, occluded, in-view for points X in each frame."""
    H, W = d["H"], d["W"]
    out = np.zeros((4, len(frames), len(X)), bool)
    for f, i in enumerate(frames):
        u, v, z = project(d, X, i)
        ok = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        sel = np.where(ok)[0]
        ui, vi = u[sel].astype(int), v[sel].astype(int)
        out[3, f, sel] = True
        g = good[i][vi, ui]
        sel, ui, vi = sel[g], ui[g], vi[g]
        r = d["depth"][i][vi, ui] - z[sel]
        tau = tau_rel * z[sel] + voxel
        out[0, f, sel[r > tau]] = True
        out[1, f, sel[np.abs(r) <= tau]] = True
        out[2, f, sel[r < -tau]] = True
    return out


def clusters_of(idx, mask, min_size):
    """Group voxel indices idx[mask] into clusters, bridging gaps up to MERGE_GAP voxels."""
    if not mask.any():
        return []
    lo_i = idx[mask].min(0)
    grid = np.zeros(idx[mask].max(0) - lo_i + 1, bool)
    grid[tuple((idx[mask] - lo_i).T)] = True
    lab, _ = ndimage.label(ndimage.binary_dilation(grid, iterations=MERGE_GAP), structure=np.ones((3, 3, 3)))
    member = np.where(mask)[0]
    labels = lab[tuple((idx[member] - lo_i).T)]
    return [member[labels == j] for j in np.unique(labels) if (labels == j).sum() >= min_size]


if __name__ == "__main__":
    d = to_room(load())
    good = good_pixels(d)
    vid, S = d["video_id"], d["S"]
    P = np.stack([unproject(d, i) for i in range(S)])

    lo = np.percentile(P[good], 0.5, axis=0)
    lo[2] = 0.0
    hi = np.percentile(P[good], 99.5, axis=0)
    dims = np.ceil((hi - lo) / VOXEL).astype(int) + 1

    def occupancy(s):
        k = np.floor((P[good & (vid == s)[:, None, None]] - lo) / VOXEL).astype(int)
        k = k[np.all((k >= 0) & (k < dims), 1)]
        occ = np.bincount(np.ravel_multi_index(k.T, dims), minlength=dims.prod()).reshape(dims) >= MIN_SUPPORT
        occ[:, :, : int(FLOOR_MARGIN / VOXEL) + 1] = False
        return occ

    occ = {s: occupancy(s) for s in (0, 1)}
    print(f"grid {dims.tolist()} at {VOXEL} m | occupied: s0 {occ[0].sum():,}, s1 {occ[1].sum():,}")

    clusters, vox = [], {}
    for kind, here, other in (("removed", 0, 1), ("added", 1, 0)):
        idx = np.argwhere(occ[here] & ~ndimage.binary_dilation(occ[other], np.ones((3, 3, 3))))
        X = lo + (idx + 0.5) * VOXEL
        see, match, occl, inview = votes(d, good, X, np.where(vid == other)[0]).sum(1)
        confirmed = (see >= MIN_SEE_THROUGH) & (see > match)
        hidden = (see + match == 0) & (inview > 0)
        unseen = inview == 0
        print(f"{kind}: {len(idx):,} candidates -> {confirmed.sum():,} confirmed, "
              f"{hidden.sum():,} hidden, {unseen.sum():,} unseen")
        for verdict, mask, min_size in ((kind, confirmed, MIN_CLUSTER),
                                        ("hidden", hidden, MIN_NOT_COMPARED), ("unseen", unseen, MIN_NOT_COMPARED)):
            for m in clusters_of(idx, mask, min_size):
                pts = X[m]
                cid = len(clusters)
                vox[f"c{cid}"] = pts.astype(np.float32)
                clusters.append(dict(
                    id=cid, verdict=verdict, present_in=here, checked_by=other, n_voxels=len(m),
                    volume_l=round(len(m) * VOXEL**3 * 1000, 1),
                    centre=np.round(pts.mean(0), 3).tolist(),
                    bbox_min=np.round(pts.min(0) - VOXEL / 2, 3).tolist(),
                    bbox_max=np.round(pts.max(0) + VOXEL / 2, 3).tolist(),
                    see_through=int(np.median(see[m])), match=int(np.median(match[m])),
                    occluded=int(np.median(occl[m])), in_view=int(np.median(inview[m]))))

    order = {"removed": 0, "added": 0, "hidden": 1, "unseen": 2}
    clusters.sort(key=lambda c: (order[c["verdict"]], -c["n_voxels"]))
    json.dump(dict(voxel=VOXEL, clusters=clusters), open("data/changes.json", "w"), indent=1)
    np.savez_compressed("data/changes_voxels.npz", **vox)

    fmt = lambda p: "(" + ", ".join(f"{x:.2f}" for x in p) + ")"
    for c in clusters:
        size = np.subtract(c["bbox_max"], c["bbox_min"])
        print(f"c{c['id']:<3} {c['verdict']:<8} in s{c['present_in']} {c['n_voxels']:>6} vox at {fmt(c['centre'])}"
              f" size {fmt(size)}  see/match/occl/in-view {c['see_through']}/{c['match']}/{c['occluded']}/{c['in_view']}")
