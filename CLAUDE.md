# Brain Tumor Segmentation — Project Memory & Roadmap

## What this project is
A PyTorch project that segments brain tumors in MRI scans using a custom 4-block-deep
U-Net (`bts/model.py: DynamicUNet`). Trained on Jun Cheng's 3064-image figshare
dataset. Achieves ~0.74 mean Dice score on a 600-image test set.

Core pieces:
- `bts/model.py` — `DynamicUNet` (U-Net variant, configurable filters, 512x512x1 in/out)
- `bts/dataset.py` — `TumorDataset` (loads `{idx}.png`/`{idx}_mask.png` pairs, random aug)
- `bts/loss.py` — `DiceLoss`, `BCEDiceLoss`
- `bts/classifier.py` — `BrainTumorClassifier` (train/test/predict/save/restore)
- `bts/plot.py` — matplotlib visualization helpers
- `api.py` — CLI inference (`--file`/`--dir` flags), loads `saved_models/UNet-[16, 32, 64, 128, 256].pt`
- `Tumor Segmentation.ipynb` — main training/eval notebook
- `setup_scripts/` — dataset download/unzip/extract
- `web/` — FastAPI web app (Phase 3, see below)
- `requirements.txt` / `requirements-web.txt` — training and web-app deps, kept separate

Owner considers this one of their best/oldest projects. No CI, no tests.

## Status: done and live
Everything below is merged into `master` and deployed. No open phases.

- **Stack**: Python 3.14.2, torch 2.13.0 / torchvision 0.28.0, FastAPI + vanilla JS
  for the web app.
- **Live app**: https://bts.subhajitdas.me — deployed via Dokploy (project "Brain
  Tumor Segmentation", app "Web"), auto-deploys on push to `master`.
- History: modernization work happened on `modernize-deps` (PR #28), the web app on
  `web-app` stacked on top (PR #29). Both merged into `master` 2026-08-09. Branches
  left in place, not deleted.

## Key decisions worth remembering
- **No retraining needed** for the dependency modernization — `state_dict` loads fine
  across the torch version jump; the shipped checkpoint is unchanged.
- **Device selection** is `cuda → mps → cpu` throughout (`api.py`, `web/inference.py`).
  This also happens to make the code Docker-portable for free: on Linux/Dokploy both
  `cuda` and `mps` are unavailable, so it cleanly falls back to `cpu` with zero
  Docker-specific code.
- **Web app** (`web/`) is a single FastAPI process serving both API routes and the
  server-rendered frontend (Jinja2 + vanilla JS, no SPA framework) — required so the
  whole thing packages into one Docker image for Dokploy.
  - Uploaded images are processed **entirely in memory**, never written to disk — no
    cleanup job needed.
  - The 12 curated "try it" sample pairs and the gallery images (WebP thumbs +
    capped full-size) are pre-baked and committed under `web/static/` by one-off
    scripts (`web/scripts/prepare_samples.py`, `web/scripts/build_gallery_thumbs.py`).
    The web app never depends on the large gitignored `dataset/` folder, and
    thumbnails aren't regenerated on every deploy.
  - Gallery source images live in `images/` at repo root, whitelisted into git via
    `.gitignore` by dice-score prefix (`!images/0.XX*`). Threshold widened from
    dice >= 0.94 to dice >= 0.85 on 2026-08-10 after syncing the full 601-image
    Google Drive results corpus (owner had it as a local zip on Desktop) — pulled in
    295 qualifying images (up from 88). Full 601-image corpus was not committed, only
    the >=0.85 subset; raw source pngs below the threshold aren't kept locally either
    (re-derive from the owner's gdrive zip if the threshold needs revisiting).
  - `bts/classifier.py` and `bts/model.py` lazily import `tensorboard`/`torchinfo`
    (scoped to `train()`/`.summary()`) rather than at module level — needed so the
    lean web image (which deliberately excludes those training-only deps) can still
    `import bts.classifier`/`bts.model` for inference.
  - Dockerfile installs CPU-only torch/torchvision via a scoped
    `--index-url https://download.pytorch.org/whl/cpu` (avoids the much larger
    default CUDA-bundled wheels). Final image: ~1.06GB.
- **Dokploy deployment**: uses the existing GitHub App integration on that instance
  (no new auth needed), builds directly from the repo's root `Dockerfile`. No
  `serverId` set — deploys on the Dokploy host itself (no separate remote servers are
  registered on this instance). Domain `bts.subhajitdas.me` → port 8000, https,
  Let's Encrypt.

## Remaining follow-ups
- Re-enable `master`'s branch protection (owner disabled the required-review rule to
  unblock merging #28/#29 — Claude can't do this itself, blocked by a permission
  classifier).
- Decide whether to delete the now-merged `modernize-deps`/`web-app` branches.
- `Tumor Segmentation.ipynb` has never been executed end-to-end for real (the code
  paths it calls have all been verified individually elsewhere).

## Working conventions for this project
- Big structural changes happen on a feature branch, not directly on `master`. Merge
  only after the owner confirms things work.
- This file should be kept up to date as decisions get made — treat it as the source
  of truth across sessions, not scratch notes. Prefer recording *decisions and
  current state* here; leave step-by-step change narration to git commit messages.
