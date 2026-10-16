"""Load a reconstruction (recon_da3.npz) and turn depth maps into world points.

Conventions: extrinsic is world-to-camera [R|t] (OpenCV: x right, y down, z forward), metres.
"""
import numpy as np

CONF_PERCENTILE = 20      # drop the lowest 20% confidence pixels
EDGE_THRESH = 0.05        # drop pixels whose relative depth gradient is above this ("drag" at object edges)


def load(path="data/recon_da3.npz"):
    d = dict(np.load(path))
    d["S"], d["H"], d["W"] = d["depth"].shape
    return d


def unproject(d, i):
    """(H,W,3) world points of frame i: X = R.T (z K^-1 [u,v,1] - t)."""
    K, E, z = d["intrinsic"][i], d["extrinsic"][i], d["depth"][i]
    v, u = np.mgrid[:d["H"], :d["W"]]
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)
    return (cam - E[:, 3]) @ E[:, :3]


def project(d, X, i):
    """World points (N,3) -> pixel coords u, v and camera depth z in frame i."""
    K, E = d["intrinsic"][i], d["extrinsic"][i]
    c = X @ E[:, :3].T + E[:, 3]
    z = c[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * c[:, 0] / z + K[0, 2]
        v = K[1, 1] * c[:, 1] / z + K[1, 2]
    return u, v, z


def good_pixels(d):
    """(S,H,W) mask of trustworthy pixels: confident, not sky, not on a depth edge."""
    depth = d["depth"]
    good = (d["conf"] > np.percentile(d["conf"], CONF_PERCENTILE)) & ~d["sky"] & (depth > 0)
    for i in range(d["S"]):
        gy, gx = np.gradient(depth[i])
        good[i] &= np.hypot(gx, gy) / np.maximum(depth[i], 1e-6) < EDGE_THRESH
    return good


def to_room(d, path="data/room_frame.npz"):
    """Re-express cameras in the room frame (room = R world + t), so unproject/project work in room coords."""
    T = np.load(path)["T"]
    R, t = T[:3, :3], T[:3, 3]
    E = d["extrinsic"].astype(np.float64)
    Ec = E[:, :, :3] @ R.T                                     # world = R.T (room - t)
    d["extrinsic"] = np.concatenate([Ec, (E[:, :, 3] - Ec @ t)[..., None]], 2)
    return d


def camera_centres(d):
    E = d["extrinsic"]
    return -np.einsum("nji,nj->ni", E[:, :, :3], E[:, :, 3])     # -R.T t
