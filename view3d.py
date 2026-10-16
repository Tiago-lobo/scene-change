"""Interactive 3D view of the room and the detected changes -> out/changes_3d.html

Grey: static scene (both sessions). Red: removed (present in session 0 only). Green: added (session 1 only).
Light blue: regions not compared (the other session never had a usable view). Diamonds: camera positions.
Usage: python view3d.py
"""
import json
import numpy as np
import plotly.graph_objects as go
from recon import load, unproject, good_pixels, to_room, camera_centres

SCENE_VOXEL = 0.04
MAX_SCENE = 150_000

d = to_room(load())
good = good_pixels(d)
p = np.concatenate([unproject(d, i)[good[i]] for i in range(d["S"])])
c = np.concatenate([d["images"][i][good[i]] for i in range(d["S"])])
_, first = np.unique(np.floor(p / SCENE_VOXEL).astype(np.int64), axis=0, return_index=True)
first = np.random.default_rng(0).permutation(first)[:MAX_SCENE]
p, c = p[first], c[first]
keep = p[:, 2] < 2.6                                              # drop the ceiling so we can look in
p, c = p[keep], c[keep]
grey = (0.35 * c.mean(1, keepdims=True) + 0.65 * 200).astype(int)  # faded colours keep context, let changes pop

traces = [go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", hoverinfo="skip", name="room",
                       marker=dict(size=1.6, color=[f"rgb({g[0]},{g[0]},{g[0]})" for g in grey]))]

ch = json.load(open("data/changes.json"))
vox = np.load("data/changes_voxels.npz")
style = {"removed": ("#d62728", 3), "added": ("#2ca02c", 3), "hidden": ("#9ecae1", 1.5), "unseen": ("#9ecae1", 1.5)}
shown = set()
for cl in ch["clusters"]:
    X = vox[f"c{cl['id']}"]
    col, size = style[cl["verdict"]]
    group = cl["verdict"] if cl["verdict"] in ("removed", "added") else "not compared"
    traces.append(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="markers", marker=dict(size=size, color=col, opacity=0.9 if size > 2 else 0.25),
        name=group, legendgroup=group, showlegend=group not in shown,
        hovertext=f"c{cl['id']} {cl['verdict']} {cl['volume_l']} L | see-through {cl['see_through']}, match {cl['match']}",
        hoverinfo="text"))
    shown.add(group)
    if cl["verdict"] in ("removed", "added"):
        top = X[X[:, 2].argmax()]
        traces.append(go.Scatter3d(x=[top[0]], y=[top[1]], z=[top[2] + 0.12], mode="text", text=[f"c{cl['id']}"],
                                   textfont=dict(color=col, size=11), showlegend=False, hoverinfo="skip"))

cams = camera_centres(d)
for s, col in ((0, "#1f77b4"), (1, "#9467bd")):
    cc = cams[d["video_id"] == s]
    traces.append(go.Scatter3d(x=cc[:, 0], y=cc[:, 1], z=cc[:, 2], mode="markers", name=f"camera, session {s}",
                               marker=dict(size=4, color=col, symbol="diamond")))

fig = go.Figure(traces)
fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                             camera=dict(eye=dict(x=-0.6, y=-1.3, z=1.1))),
                  margin=dict(l=0, r=0, t=30, b=0), title="Detected changes (room frame, metres)",
                  legend=dict(itemsizing="constant"))
fig.write_html("out/changes_3d.html", include_plotlyjs=True)
print("out/changes_3d.html")
