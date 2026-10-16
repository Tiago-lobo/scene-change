# What changed in the room?

Given two phone videos of the same room taken at different times, this pipeline reconstructs a shared 3D map of both sessions, finds the voxels where something changed, and describes each change in words — what the object is, whether it looks like a real change or alignment noise, and whether a removed thing is the same as an added thing somewhere else.

- Geometry (stages 1-3) is deterministic 3D work: reconstruction + a visibility-aware voxel diff.
- Semantics (stage 4) runs **only** where geometry already flagged a change ("steered segmentation"), so a vision-model false positive cannot invent a change.
- The default semantics path calls **Claude Haiku 4.5** to describe and reason about each flagged region. An offline fallback using DINOv2 + BLIP is included for runs without an API key.

Walkthrough images in [`showcase/`](showcase/) show each step with real output from the sample run.

## Example results

Two sessions of a classroom (session A ≈ 22 s, session B ≈ 24 s, iPhone 13 Pro Max 4K/30 fps). Stage 3 flagged 15 candidate change regions; stage 4 turned those into the following findings. Every quote below is the actual VLM response in `data/vlm.json`, not a summary.

### MOVED — blue speaker cube, 2.61 m (high confidence)

Cluster c7 (`out/vlm_crops/c7_f472_crop.png`) was removed in session B at `(7.71, 2.06, 0.91)` with 10/10 empty-votes; cluster c18 (`out/vlm_crops/c18_f624_crop.png`) was added in session B at `(5.14, 1.61, 0.88)` with 12/22 empty-votes.

> VLM describe (c7): *"blue jewelry box or small branded container"* — `real_change` conf 0.85.
> VLM describe (c18): *"blue plastic container or box"* — `real_change` conf 0.85.
> VLM pair: *"Both crops show an identical blue cubic box with the same proportions, color, and apparent branding/label on the front face, consistent with being the same physical object photographed from different angles and distances."* — `same_object` conf 0.85.

### MOVED — garment, 0.37 m (high confidence)

Cluster c3 removed at `(6.47, 2.9, 0.86)` with 24/24 empty-votes; cluster c20 added at `(6.24, 3.19, 0.89)` with 14/14 empty-votes.

> VLM pair: *"Both crops show the same olive/khaki colored tactical or military-style garment with similar fabric texture and fold patterns, consistent with being the same item photographed from different angles and distances."* — `same_object` conf 0.85.

### MISALIGNMENT — monitor station, 0.27 m (low confidence, not reported as a move)

> VLM pair: *"Both show an identical black flat-screen monitor mounted on what appears to be the same desk setup, with consistent positioning and styling across the two viewpoints."* — `same_object` conf 0.92.

Distance is below the 30 cm move threshold, so this is reported as "likely alignment noise" rather than a move.

### ADDED — white roller blind on the window (medium confidence)

Cluster c17, 6/22 empty-votes.

> VLM: *"A white roller blind appears to be present at the bottom of the window frame, suggesting it was added between the two viewpoints."* — `real_change` conf 0.85.

### SUPPRESSED — six small clusters the VLM flagged as noise

For example cluster c0:

> *"The stripe pattern appears to be a consistent part of the surface, and the visible edge shift is likely due to camera perspective change rather than actual removal."* — `alignment_artefact` conf 0.70.

Each suppressed cluster has a similar reasoning string in `data/vlm.json`.

**Confidence levels** combine three independent signals: how many frames of the other session saw past the spot (`see_through / in_view`), how far the crop sits from the frame edge, and the VLM's own confidence in its pair/description decision. "High" means at least 80 % empty-votes, a reasonable crop (no frame-edge clipping), and a VLM pair verdict of `same_object` at ≥ 0.7. "Low" is any of those failing. "Medium" is in-between.

Open `out/report.md` for the full human-readable report, `out/changes_3d_vlm.html` for an interactive 3D view, and `data/vlm.json` for the structured output.

## How to run

### Quickest path — Google Colab

Open `notebooks/run_pipeline_colab.ipynb` in Colab and run the cells top to bottom. The notebook installs dependencies, points at a folder you provide (via Drive mount, upload, or GitHub clone), runs stages 2-4 against the sample `data/recon_da3.npz`, and shows the report, the interactive 3D view, and the walkthrough slides inline. A GPU runtime isn't required.

### Local run

#### Requirements

- Python 3.12 and the packages in `requirements.txt` (`pip install -r requirements.txt`). PyTorch is used in CPU, MPS, or CUDA mode depending on what the machine has.
- A Kaggle account (or any machine with ~16 GB VRAM) to run stage 1 once. The other stages run on a laptop.
- For the primary semantics path, an Anthropic API key exported as `ANTHROPIC_API_KEY`. For the offline fallback, no key is needed.

### Inputs

Two phone videos of the same room from different viewpoints. The sample data in `data/recon_da3.npz` is 46 keyframes (22 from session A, 24 from session B) at 280×504 px, extracted and depth-mapped by stage 1.

### Stage 1 — shared 3D reconstruction

Run `notebooks/stage1_reconstruct_da3_kaggle.ipynb` on Kaggle. It picks sharp keyframes from both videos, runs Depth Anything 3 (metric variant) on all of them in a single pass, and writes `data/recon_da3.npz` with images, per-pixel depth, confidence, sky masks, intrinsics and extrinsics for every frame — all in one shared metric coordinate frame. Everything from here runs locally against that file.

### Stage 2 — room frame

Fits the floor plane, snaps the walls to the x/y axes, and places the origin at the floor corner so all coordinates are positive and `z` is up.

```sh
python room_frame.py
```

### Stage 3 — visibility-aware voxel diff

Voxelises both sessions at 3 cm. A voxel becomes a candidate change if it is occupied in one session with nothing within one voxel in the other. For each candidate, the pipeline projects it into every frame of the other session and counts three visibility states: *see-through* (depth passed through the spot), *match* (something is there), and *occluded* (something in front). A candidate is confirmed if see-through votes are ≥ 3 and dominate match. Voxels that cannot be checked (nobody looked at them, or always occluded) are reported honestly as "not compared" rather than treated as changes.

```sh
python diff.py
python show_clusters.py   # optional: writes out/clusters_*.png evidence sheets
```

### Stage 4 — semantics (primary path, Claude Haiku 4.5)

For every removed/added cluster, projects the cluster's voxels into the owning session's best frame and uses them as a point prompt for SAM. SAM returns a tight mask of the object, which is cropped and sent to Claude Haiku 4.5. Haiku describes what is in the crop and classifies the change as `real_change`, `alignment_artefact`, or `uncertain`. For every pair of candidate `(removed, added)` clusters inside a 6 m radius, Haiku is also asked whether they are `same_object`, `similar_but_distinct` or `unrelated`, with a short reason. Final move pairing is greedy best-first, scored by `vlm_confidence × exp(-distance / 4 m)` so that close same-object matches dominate. Pairs under 30 cm are tagged as likely alignment rather than moves.

```sh
export ANTHROPIC_API_KEY=sk-ant-...
python vlm_reason.py
```

### Stage 4 — semantics (offline fallback)

Uses SAM for masks, DINOv2 for crop embeddings and BLIP for captions. No API key required. Produces the same JSON structure in `data/semantics.json` with weaker captions and no explicit significance reasoning.

```sh
python semantics.py
```

### 3D visualisation

```sh
python view3d_vlm.py       # primary pipeline -> out/changes_3d_vlm.html
python view3d.py           # fallback pipeline -> out/changes_3d.html
```

Open the generated HTML in any browser. Hover any dot or arrow to see the VLM's description and reasoning.

### Outputs you actually look at

- `out/report.md` — human-readable report with one section per finding.
- `out/changes_3d_vlm.html` — interactive 3D view of the changes over the pooled scene cloud.
- `out/vlm_crops/c<id>_f<frame>_crop.png` — per-cluster image crops the VLM reasoned about.
- `out/clusters_*.png` — stage 3 geometric evidence sheets.

Everything else in `data/` is intermediate and reproducible from the scripts.

## Design notes

- **Geometry first, semantics after.** The vision model never generates a change. It only describes, judges, and matches changes that the voxel diff has already localised to a 3 cm region. False positives in detection or captioning cannot invent findings; they can only be filtered out as noise.
- **Visibility-aware diff.** Every candidate voxel asks every frame of the other session what it saw: see-through, match, occluded, out-of-view. Only confirmed voxels become clusters; voxels no camera could verify are reported as "not compared" rather than silently treated as unchanged.
- **Steered segmentation.** For each flagged region we project its 3D voxels into the owning session's best frame and feed those points to SAM as a *point prompt*. SAM returns a precise object boundary derived from the geometry, so the crop is tight to the object instead of a loose box full of surrounding furniture.
- **Scene-agnostic prompts.** The VLM prompts never name any object category — no "chair", "monitor", "window". They mention only *physical space*, *outlined region*, and the three generic verdicts. The pipeline works on any room.
- **Scoring.** Pair selection combines VLM confidence with a distance discount; a confirmed pair closer than 30 cm is demoted to "likely alignment noise" rather than a move. Confidence levels combine the geometry evidence (empty-vote ratio), VLM reasoning, and crop quality.
- **Deterministic report.** The report is built from the structured JSON, not from a free-form LLM write-up. Every claim in it traces back to a cluster id, a crop file, and the exact VLM response that justified it.

## Attribution

- **Depth Anything 3** (ByteDance Seed) — monocular metric depth with camera intrinsics and extrinsics, in one shared frame across all input videos. The nested giant checkpoint used here is CC BY-NC 4.0.
- **SAM 2.1** (Meta) — pixel-exact object masks from box/point prompts.
- **SuperMap** (Zhao et al., arXiv 2608.22896) — the three-state visibility test, multi-frame evidence, and semantic consistency idea.
- **DINOv2** and **BLIP** (Meta / Salesforce) — the offline semantics fallback.
- **Claude Haiku 4.5** (Anthropic) — the VLM reasoning in the primary stage 4.
