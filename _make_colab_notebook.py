"""One-shot helper: build notebooks/run_pipeline_colab.ipynb from inline cells.

Not a stage of the pipeline. Run once after editing the cells here to refresh the
generated notebook, then commit the notebook.
"""
import json, os

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.splitlines(keepends=True)})


md("""# Scene change detection — run end-to-end in Colab

This notebook runs the full "what changed in the room?" pipeline against a project folder you provide.

**What you need:**
- A folder (uploaded to Colab or mounted from Drive) containing the project source files and `data/recon_da3.npz` (the shared 3D reconstruction from stage 1).
- An Anthropic API key for the stage-4 VLM reasoning (free Haiku 4.5 trial credit is enough).
- A GPU runtime is optional; everything also runs on CPU.

**What it does:**
1. Installs dependencies
2. Points at your project folder
3. Builds the room frame (stage 2)
4. Finds change clusters with the visibility-aware voxel diff (stage 3)
5. Runs Claude Haiku 4.5 on each flagged region to describe and reason about it (stage 4)
6. Shows you the report, 3D view, and showcase slides inline.
""")

md("""## 1. Install dependencies""")

code("""!pip install -q torch transformers anthropic plotly scipy pillow matplotlib""")

md("""## 2. Point at your project folder

Pick **one** of the options below (comment the other two out). Each puts you in a working directory that contains `recon.py`, `vlm_reason.py`, etc. and a `data/recon_da3.npz` file.
""")

code("""# --- Option A: clone from a public GitHub repo (fastest if the repo is public) ---
# !git clone -q https://github.com/YOUR_USER/scene-change
# %cd scene-change

# --- Option B: upload a zipped project folder ---
# from google.colab import files
# uploaded = files.upload()        # drop your scene-change.zip here
# !unzip -q scene-change.zip -d .
# %cd scene-change

# --- Option C: mount Google Drive (recommended for repeated runs) ---
from google.colab import drive
drive.mount('/content/drive')
PROJECT_PATH = '/content/drive/MyDrive/scene-change'  # <-- edit if your folder lives elsewhere
%cd {PROJECT_PATH}
""")

code("""# Sanity check: the sample reconstruction must be present.
import os
assert os.path.exists('data/recon_da3.npz'), (
    "data/recon_da3.npz not found. If you want to generate it yourself, "
    "run notebooks/stage1_reconstruct_da3_kaggle.ipynb once on Kaggle (needs a T4-class GPU)."
)
print('Project folder OK. Files on disk:')
!ls -1 *.py && echo --- && ls -1 data/
""")

md("""## 3. Stage 2 — put the reconstruction into a room frame

Fits the floor plane, snaps the walls to the x/y axes, and places the origin at the floor corner so all coordinates are positive and `z` is up. Writes `data/room_frame.npz`.""")

code("""!python room_frame.py""")

md("""## 4. Stage 3 — visibility-aware voxel diff

Voxelises both sessions at 3 cm, finds candidate change voxels (occupied in one session, empty in the other), and runs a per-voxel visibility test against every frame of the opposite session. Only confirmed change voxels become clusters. Writes `data/changes.json` + `data/changes_voxels.npz`, plus `out/clusters_*.png` evidence sheets.""")

code("""!python diff.py""")

code("""!python show_clusters.py
from IPython.display import Image, display
for p in sorted([p for p in __import__('os').listdir('out') if p.startswith('clusters_')]):
    display(Image(filename=f'out/{p}'))
""")

md("""## 5. Stage 4 — VLM semantics via Claude Haiku 4.5

For every removed/added cluster, projects its voxels into the owning session's best frame, uses them as a SAM point prompt to get a tight object mask, crops around it, and asks Claude Haiku 4.5 to describe and classify the change. Candidate `(removed, added)` pairs are then compared pairwise for identity.""")

code("""# Securely enter your Anthropic API key (not stored on disk, not echoed back).
import getpass, os
os.environ['ANTHROPIC_API_KEY'] = getpass.getpass('Anthropic API key: ')
""")

code("""!python vlm_reason.py""")

md("""## 6. Results

The structured output is in `data/vlm.json`. Here is the human-readable report and the interactive 3D view.""")

code("""from IPython.display import Markdown
Markdown(open('out/report.md').read())""")

code("""!python view3d_vlm.py
from IPython.display import HTML
HTML(open('out/changes_3d_vlm.html').read())""")

md("""## 7. Walkthrough slides

Seven PNGs that trace one example from input frames through to the final map of findings. Useful for building a video or presentation.""")

code("""!python make_showcase.py
from IPython.display import Image, display
import os
for p in sorted(os.listdir('showcase')):
    if p.endswith('.png'):
        print(p)
        display(Image(filename=f'showcase/{p}'))
""")

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

os.makedirs("notebooks", exist_ok=True)
path = "notebooks/run_pipeline_colab.ipynb"
with open(path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {path} ({len(cells)} cells)")
