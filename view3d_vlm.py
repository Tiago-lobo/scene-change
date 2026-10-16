"""Interactive 3D view of the VLM pipeline's output -> out/changes_3d_vlm.html

Grey faded point cloud: the room, for context.
Red dots: removed singletons (cluster voxels at the "from" spot of a VLM-confirmed removal).
Green dots: added singletons.
Orange arrows with labels: confirmed moves (VLM pair said same_object, distance > MOVE_MIN).
Olive dashed arrows: likely-misalignment pairs (VLM pair said same_object but distance < MOVE_MIN).
Pink dashed arrows: VLM similar_but_distinct close pairs ("possible same, uncertain").
Faded purple dots: clusters the VLM suppressed as alignment artefacts.
Camera diamonds: session 0 (blue) and session 1 (purple).

Every cluster hovers to show the VLM's description and significance verdict.
Usage: python view3d_vlm.py
"""
import json
import numpy as np
import plotly.graph_objects as go
from recon import load, unproject, good_pixels, to_room, camera_centres

SCENE_VOXEL = 0.04
MAX_SCENE = 150_000


def add_cluster(fig, pts, colour, size, opacity, group, shown, hover, legend_name=None):
    fig.add_scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                      marker=dict(size=size, color=colour, opacity=opacity),
                      name=legend_name or group, legendgroup=group,
                      showlegend=group not in shown,
                      hovertext=hover, hoverinfo="text")
    shown.add(group)


def label_at(pts, text, colour, size=11):
    top = pts[pts[:, 2].argmax()]
    return go.Scatter3d(x=[top[0]], y=[top[1]], z=[top[2] + 0.12],
                        mode="text", text=[text],
                        textfont=dict(color=colour, size=size),
                        showlegend=False, hoverinfo="skip")


def arrow(fig, A, B, colour, dash, group, shown, hover):
    fig.add_scatter3d(x=[A[0], B[0]], y=[A[1], B[1]], z=[A[2] + 0.05, B[2] + 0.05],
                      mode="lines", line=dict(color=colour, width=6, dash=dash),
                      name=group, legendgroup=group, showlegend=group not in shown,
                      hovertext=hover, hoverinfo="text")
    shown.add(group)
    v = np.subtract(B, A); n = np.linalg.norm(v) or 1
    fig.add_cone(x=[B[0]], y=[B[1]], z=[B[2] + 0.05],
                 u=[v[0]/n], v=[v[1]/n], w=[v[2]/n],
                 sizemode="absolute", sizeref=0.25, anchor="tip", showscale=False,
                 colorscale=[[0, colour], [1, colour]],
                 hoverinfo="skip", showlegend=False)


if __name__ == "__main__":
    d = to_room(load())
    good = good_pixels(d)
    vlm = json.load(open("data/vlm.json"))
    vox = np.load("data/changes_voxels.npz")

    # ----- faded context cloud -----
    p = np.concatenate([unproject(d, i)[good[i]] for i in range(d["S"])])
    c = np.concatenate([d["images"][i][good[i]] for i in range(d["S"])])
    _, first = np.unique(np.floor(p / SCENE_VOXEL).astype(np.int64), axis=0, return_index=True)
    first = np.random.default_rng(0).permutation(first)[:MAX_SCENE]
    p, c = p[first], c[first]
    keep = p[:, 2] < 2.6
    p, c = p[keep], c[keep]
    grey = (0.35 * c.mean(1, keepdims=True) + 0.65 * 200).astype(int)

    fig = go.Figure()
    fig.add_scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", hoverinfo="skip",
                      name="room",
                      marker=dict(size=1.6,
                                  color=[f"rgb({g[0]},{g[0]},{g[0]})" for g in grey]))
    shown = {"room"}

    def pts_for(cid):
        key = f"c{cid}"
        return vox[key] if key in vox.files else None

    # ----- confirmed moves (orange arrow + dot clouds at each end) -----
    for m in vlm["moves"]:
        f, t = m["from_"], m["to"]
        pa, pb = pts_for(f["id"]), pts_for(t["id"])
        hover_a = f"c{f['id']} FROM: {f['description']}<br>VLM: {f.get('vlm_why','')}"
        hover_b = f"c{t['id']} TO: {t['description']}<br>VLM: {t.get('vlm_why','')}"
        if pa is not None:
            add_cluster(fig, pa, "#ff7f0e", 3, 0.9, "move", shown, hover_a)
            fig.add_trace(label_at(pa, f"c{f['id']}: {f['description'][:40]}", "#ff7f0e"))
        if pb is not None:
            add_cluster(fig, pb, "#ff7f0e", 3, 0.9, "move", shown, hover_b)
            fig.add_trace(label_at(pb, f"c{t['id']}: {t['description'][:40]}", "#ff7f0e"))
        A = np.array(f["centre"]); B = np.array(t["centre"])
        label = f"{m['pair_why'][:80]} ({m['distance_m']} m, {m['confidence']})"
        arrow(fig, A, B, "#ff7f0e", "solid", "move", shown, label)
        mid = (A + B) / 2
        fig.add_scatter3d(x=[mid[0]], y=[mid[1]], z=[mid[2] + 0.3], mode="text",
                          text=[f"MOVE {m['distance_m']} m"],
                          textfont=dict(color="#ff7f0e", size=12),
                          showlegend=False, hoverinfo="skip")

    # ----- misalignments (olive dashed) -----
    for m in vlm.get("misalignments", []):
        f, t = m["from_"], m["to"]
        pa, pb = pts_for(f["id"]), pts_for(t["id"])
        hover_a = f"c{f['id']} FROM: {f['description']}"
        hover_b = f"c{t['id']} TO: {t['description']}"
        if pa is not None:
            add_cluster(fig, pa, "#bcbd22", 3, 0.85, "misalignment", shown, hover_a)
        if pb is not None:
            add_cluster(fig, pb, "#bcbd22", 3, 0.85, "misalignment", shown, hover_b)
        arrow(fig, np.array(f["centre"]), np.array(t["centre"]), "#bcbd22", "dash",
              "misalignment", shown, m["pair_why"][:80])

    # ----- uncertain pairs (pink dashed) -----
    for p_ in vlm.get("uncertain_pairs", []):
        f, t = p_["from_"], p_["to"]
        pa, pb = pts_for(f["id"]), pts_for(t["id"])
        if pa is not None:
            add_cluster(fig, pa, "#e377c2", 3, 0.75, "uncertain pair", shown,
                        f"c{f['id']} (uncertain): {f['description']}")
        if pb is not None:
            add_cluster(fig, pb, "#e377c2", 3, 0.75, "uncertain pair", shown,
                        f"c{t['id']} (uncertain): {t['description']}")
        arrow(fig, np.array(f["centre"]), np.array(t["centre"]), "#e377c2", "dot",
              "uncertain pair", shown, p_["pair_why"][:80])

    # ----- singletons (removed red, added green) -----
    for s in vlm["singletons"]:
        pp = pts_for(s["id"])
        if pp is None:
            continue
        if s["verdict"] == "removed":
            col, group = "#d62728", "removed (singleton)"
        else:
            col, group = "#2ca02c", "added (singleton)"
        hover = (f"c{s['id']} {s['verdict']}: {s['description']}<br>"
                 f"VLM: {s['significance']} ({s['vlm_confidence']:.2f}) - {s['vlm_why']}")
        add_cluster(fig, pp, col, 3, 0.9, group, shown, hover)
        fig.add_trace(label_at(pp, f"c{s['id']}: {s['description'][:30]}", col, size=10))

    # ----- VLM-suppressed artefacts (faded purple dots, no label) -----
    for a in vlm.get("artefacts", []):
        pp = pts_for(a["id"])
        if pp is None:
            continue
        hover = (f"c{a['id']} (suppressed): {a['description']}<br>"
                 f"VLM: {a['significance']} ({a['vlm_confidence']:.2f}) - {a['vlm_why']}")
        add_cluster(fig, pp, "#9467bd", 2, 0.35, "suppressed artefact", shown, hover)

    # ----- camera positions -----
    cams = camera_centres(d)
    for s, col in ((0, "#1f77b4"), (1, "#9467bd")):
        cc = cams[d["video_id"] == s]
        fig.add_scatter3d(x=cc[:, 0], y=cc[:, 1], z=cc[:, 2], mode="markers",
                          name=f"camera, session {s}",
                          marker=dict(size=4, color=col, symbol="diamond"))

    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x (m)",
                                 yaxis_title="y (m)", zaxis_title="z (m)",
                                 camera=dict(eye=dict(x=-0.6, y=-1.3, z=1.1))),
                      margin=dict(l=0, r=0, t=30, b=0),
                      title="VLM pipeline (geometry + Claude Haiku 4.5): detected changes",
                      legend=dict(itemsizing="constant"))
    fig.write_html("out/changes_3d_vlm.html", include_plotlyjs=True)
    print("out/changes_3d_vlm.html")
