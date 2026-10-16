"""Stage 2: put the reconstruction into a room frame.

  z = up, floor at z = 0, walls parallel to x and y, x along the room's long side, origin at the
  floor's corner so all coordinates are positive.

1. Up: the biggest roughly horizontal plane (RANSAC; here the desk tops) gives the up direction.
2. Floor: the lowest strong peak in the histogram of point heights.
3. Walls: rotate about up and pick the angle where points pile up most sharply along x and y
   (lowest histogram entropy) - walls become thin spikes when aligned with an axis.

Writes data/room_frame.npz with T (4x4, world -> room). Usage: python room_frame.py
"""
import numpy as np
from recon import load, unproject, good_pixels, camera_centres

SAMPLE_VOXEL = 0.03       # downsample before fitting (m)
PLANE_TOL = 0.02          # RANSAC inlier distance (m)
MAX_TILT = 30             # candidate planes must be within this many degrees of the cameras' "up"


def sample_points(d, good):
    p = np.concatenate([unproject(d, i)[good[i]] for i in range(d["S"])])
    _, first = np.unique(np.floor(p / SAMPLE_VOXEL).astype(np.int64), axis=0, return_index=True)
    return p[first]


def fit_up(p, up0, iters=3000, rng=np.random.default_rng(0)):
    best, best_n, best_a = 0, None, None
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n) * np.sign(n @ up0)
        if n @ up0 < np.cos(np.radians(MAX_TILT)):
            continue
        k = np.sum(np.abs((p - a) @ n) < PLANE_TOL)
        if k > best:
            best, best_n, best_a = k, n, a
    q = p[np.abs((p - best_a) @ best_n) < PLANE_TOL]
    n = np.linalg.svd(q - q.mean(0), full_matrices=False)[2][2]       # refit on inliers
    return n * np.sign(n @ up0)


def floor_height(h, bin=0.02, min_share=0.01):
    """Lowest histogram peak holding at least min_share of all points."""
    hist, edges = np.histogram(h, bins=np.arange(np.percentile(h, 0.1), np.percentile(h, 99.9), bin))
    strong = np.where(hist >= min_share * len(h))[0]
    i = strong[0]
    while i + 1 < len(hist) and hist[i + 1] > hist[i]:                 # climb to the top of that peak
        i += 1
    return edges[i] + bin / 2


def wall_angle(xy, bin=0.03):
    def entropy(v):
        hist = np.histogram(v, bins=np.arange(v.min(), v.max() + bin, bin))[0]
        pr = hist[hist > 0] / len(v)
        return -(pr * np.log(pr)).sum()
    angles = np.radians(np.arange(0, 90, 0.25))
    cost = [entropy(xy @ [np.cos(a), np.sin(a)]) + entropy(xy @ [-np.sin(a), np.cos(a)]) for a in angles]
    return angles[int(np.argmin(cost))]


if __name__ == "__main__":
    d = load()
    p = sample_points(d, good_pixels(d))
    up0 = -d["extrinsic"][:, 1, :3].mean(0)                            # cameras' y axis points down
    up = fit_up(p, up0 / np.linalg.norm(up0))

    # orthonormal basis (e1, e2, up), then rotate e1/e2 to the walls
    e1 = np.cross(up, [0, 0, 1.0]) if abs(up[2]) < 0.9 else np.cross(up, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    xy = np.stack([p @ e1, p @ e2], 1)
    a = wall_angle(xy)
    x_ax, y_ax = np.cos(a) * e1 + np.sin(a) * e2, -np.sin(a) * e1 + np.cos(a) * e2
    ext = lambda ax: np.percentile(p @ ax, 99) - np.percentile(p @ ax, 1)
    if ext(y_ax) > ext(x_ax):                                          # x along the long side
        x_ax, y_ax = y_ax, -x_ax
    R = np.stack([x_ax, y_ax, up])                                     # rows = room axes in world coords

    q = p @ R.T
    origin = np.array([np.percentile(q[:, 0], 1), np.percentile(q[:, 1], 1), floor_height(q[:, 2])])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -origin
    np.savez("data/room_frame.npz", T=T)

    q = q - origin
    h = q[:, 2]
    cams = camera_centres(d) @ R.T - origin
    print(f"up is {np.degrees(np.arccos(np.clip(up @ up0 / np.linalg.norm(up0), -1, 1))):.1f} deg from the cameras' average up")
    print(f"room: x {np.percentile(q[:, 0], 99):.2f} m, y {np.percentile(q[:, 1], 99):.2f} m, ceiling ~{np.percentile(h, 99.5):.2f} m")
    hist, edges = np.histogram(h, bins=np.arange(0, 1.2, 0.02))
    desk = edges[np.argmax(hist * (edges[:-1] > 0.4))] + 0.01
    print(f"desk tops at {desk:.2f} m (expect ~0.75) | camera height {cams[:, 2].mean():.2f} m")
    for s in (0, 1):
        c = cams[d["video_id"] == s].mean(0)
        print(f"session {s} stood at ({c[0]:.2f}, {c[1]:.2f})")
