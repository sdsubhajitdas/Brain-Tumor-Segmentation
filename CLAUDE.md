# Brain Tumor Segmentation — Project Memory & Roadmap

## What this project is
A 2019-era PyTorch project that segments brain tumors in MRI scans using a custom
4-block-deep U-Net (`bts/model.py: DynamicUNet`). Trained on Jun Cheng's 3064-image
figshare dataset. Achieves ~0.74 mean Dice score on a 600-image test set.

Core pieces:
- `bts/model.py` — `DynamicUNet` (U-Net variant, configurable filters, 512x512x1 in/out)
- `bts/dataset.py` — `TumorDataset` (loads `{idx}.png`/`{idx}_mask.png` pairs, random aug)
- `bts/loss.py` — `DiceLoss`, `BCEDiceLoss`
- `bts/classifier.py` — `BrainTumorClassifier` (train/test/predict/save/restore)
- `bts/plot.py` — matplotlib visualization helpers
- `api.py` — CLI inference (`--file`/`--dir` flags), loads `saved_models/UNet-[16, 32, 64, 128, 256].pt`
- `Tumor Segmentation.ipynb` — main training/eval notebook
- `setup_scripts/` — dataset download/unzip/extract
- `requirements.txt` — pinned to 2019 versions (torch==1.0.1, torchvision==0.2.2, etc.)

Owner considers this one of their best/oldest projects. Repo is otherwise untouched —
no CI, no tests, README badge says "Maintenance: No".

## The plan (4 phases, roughly sequential)

### Phase 1 — Isolated environment for the original code
Goal: don't let this project's deps clobber other Python/PyTorch installs on the
owner's machine. Use a Python **virtual environment** (`python3 -m venv`), same idea
as a project-local `node_modules`.

**Status: blocked, needs a decision.** Verified on this machine (macOS, Apple Silicon
arm64, system Python 3.14, no conda/pyenv): PyPI has no `torch==1.0.1` wheel for this
platform/Python combo at all — pip only resolves down to torch 2.9.0 here. The original
pinned stack (torch 1.0.1 / torchvision 0.2.2 / Python era ~3.6-3.7) is not realistically
installable as-is on this machine (no arm64 macOS wheels existed for torch that old;
would need an x86_64 Python under Rosetta plus an ancient Python interpreter — high
effort, fragile, and about to be thrown away by Phase 2 anyway).

**Decision (confirmed by owner 2026-08-09):** skip trying to resurrect the exact 2019
stack. Go straight to a venv with a current Python + latest compatible deps — this
collapses Phase 1 and Phase 2 into one step (isolate now, on the versions we're
migrating to anyway).

### Phase 2 — Migrate to latest Python + dependencies
**Status: done, verified working. Not yet committed/merged (owner said "we will push
over there" — commit is pending owner go-ahead).**

- Branch: `modernize-deps` (created off `master`, `master` untouched).
- `.venv` created at project root using the system Python already on this machine
  (Python 3.14.2, macOS arm64). Not committed (already in `.gitignore`).
- `requirements.txt` updated to latest versions as of 2026-08-09: torch 2.13.0,
  torchvision 0.28.0, numpy 2.5.1, matplotlib 3.11.1, Pillow 12.3.0, requests 2.34.2,
  tensorboard 2.21.0, torchinfo 1.8.0, tqdm 4.70.0, h5py 3.16.0, jupyterlab 4.6.2.
  (`torchsummary`/`tensorboardX` dropped — replaced, see below.)
- Code fixes applied:
  - `bts/model.py`: `F.sigmoid` → `torch.sigmoid`; `.summary()` now uses `torchinfo`
    instead of unmaintained `torchsummary` (signature: `input_size` no longer includes
    batch dim implicitly — now built as `(batch_size, *input_size)`).
  - `bts/classifier.py`: `tensorboardX.SummaryWriter` → `torch.utils.tensorboard.SummaryWriter`;
    dropped unused `from torch.autograd import Variable`; fixed a real bug in
    `restore_model` (was comparing `self.device == 'cpu'` against a `torch.device`
    object, which is never true — always took the "else" branch regardless of device).
    Now just `torch.load(path, map_location=self.device)` unconditionally.
  - `api.py`: device selection now checks `cuda` → `mps` → `cpu` (was cuda/cpu only,
    missed Apple Silicon GPU acceleration entirely). Also fixed a pre-existing bug:
    when `--ofp`/`--odp` was given for a single `--file`, the output path kept the
    full input directory structure instead of just the basename, so saving failed
    with `FileNotFoundError` unless the exact nested folder structure existed under
    the output dir. Fixed with `os.path.basename` + `os.path.splitext`.
- **Verified:** existing checkpoint (`saved_models/UNet-[16, 32, 64, 128, 256].pt`)
  loads without any conversion step on the new torch version, and `api.py` produces
  a predicted mask for a sample image that visually matches the ground-truth mask's
  location/shape (consistent with the original ~0.74 Dice score). **No retraining was
  needed for this migration**, confirming the earlier prediction.
- Not yet touched: `Tumor Segmentation.ipynb` still has cuda-only device selection
  (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`) and imports
  `bts.model`/`bts.classifier` the old way — should still work since the underlying
  modules were fixed, but the notebook itself hasn't been re-run end-to-end on the
  new stack yet, and its device selection doesn't get MPS acceleration.
- Cross-checked against the project's GitHub issues (github.com/sdsubhajitdas/Brain-Tumor-Segmentation/issues):
  - #14 (`F.sigmoid` deprecated) and #18 (`torchsummary`→`torchinfo`) — already fixed above.
  - #6 (closed) — owner had already agreed the `restore_model` device bug was real;
    matches the fix made above independently.
  - #27 (open, `np.resize` silently corrupts on shape mismatch) — fixed: replaced all
    6 occurrences (`bts/classifier.py` x5, `api.py` x1) with `.reshape()`. Every call
    site always had matching element counts (single-channel 512x512 data), so this is
    a pure safety/perf improvement with no behavior change, not a functional fix.
  - #8 (open, `TypeError` crash rotating grayscale images — a Pillow/torchvision
    `fillcolor` incompatibility) — verified it no longer reproduces with
    torchvision 0.28 / Pillow 12.3. Resolved as a side effect of the dependency bump,
    no code change needed. Did not comment/close on GitHub — that's an owner call.
- `Tumor Segmentation.ipynb` cell 2 (device selection) updated: cuda → mps → cpu
  fallback, and the device-name print no longer unconditionally calls
  `torch.cuda.get_device_name()` (which would crash on non-CUDA machines). Verified
  the equivalent logic standalone; did not execute the full notebook since the
  `dataset/` folder isn't present locally (needs `setup_scripts/download_dataset.py`
  first, a large multi-GB download — not yet done).
- `README.md` installation section updated: added `.venv` creation/activation steps,
  clarified that `requirements.txt` now installs a CPU/MPS-capable PyTorch build by
  default and CUDA users should install a matching build separately.
- `setup_scripts/` (`download_dataset.py`, `unzip_dataset.py`, `extract_images.py`)
  reviewed and **run end-to-end for real, on this machine — no code changes needed**.
  None of the three import torch/torchvision, so they were never touched by the
  PyTorch migration; their only deps are requests/tqdm (download), stdlib `zipfile`
  (unzip), and h5py/numpy/matplotlib (extract), all already pinned to current
  versions in `requirements.txt`.
  - Ran `python setup_scripts/download_dataset.py` — all 4 figshare zip parts
    (~880MB total) + README downloaded successfully into `dataset/` and renamed to
    `*_done.zip` as the script expects (the URLs are presigned S3 links, so a plain
    `HEAD` check 403s per-method, but `GET`/streamed download works fine).
  - Ran `python setup_scripts/unzip_dataset.py` — extracted all 4 parts into
    `dataset/mat_dataset/`, producing exactly 3064 `.mat` files.
  - Ran `python setup_scripts/extract_images.py` — converted all 3064 `.mat` files
    into `dataset/png_dataset/` (6128 PNGs: `{idx}.png` + `{idx}_mask.png` pairs,
    0-indexed). h5py read path (`file.get('cjdata/image')`/`cjdata/tumorMask')`
    confirmed working against real data, not just import-checked.
  - Further verified the *output* is actually usable: loaded it with
    `bts.dataset.TumorDataset` (len=3064, sample tensors correctly
    `[1, 512, 512]` float32 in `[0, 1]` — the PNGs save as RGBA via
    `mpimg.imsave(..., cmap='gray')`, but `TumorDataset`'s `transforms.Grayscale()`
    already collapses that, so no code change needed there either), then ran 4 real
    samples through the trained checkpoint via `BrainTumorClassifier.predict()` on
    the `mps` device — dice scores 0.87 / 0.94 / 0.94 / 0.28, consistent spread
    around the reported ~0.74 mean over the actual held-out test set.
  - `dataset/` (~1GB) is local-only, already covered by `.gitignore`, not committed.
  - Note: this also means `Tumor Segmentation.ipynb` can now actually be re-run
    end-to-end for real (previously blocked on the dataset not existing locally) —
    still not yet done as of this writing.

### Phase 3 — Web app to host the model
- Users can either pick from example tumor images or upload their own image.
- App runs inference (reusing `bts/model.py` + `bts/classifier.py` logic, similar to
  `api.py`) and returns/display the predicted mask.
- Deployment target: owner's personal VPS, managed via **Dokploy**.
- Not yet scoped: frontend/backend stack choice, whether inference runs synchronously
  in the request or via a job queue, how example images are bundled, auth/rate-limiting
  needs for public upload.

## Open questions to resolve with the owner
- [x] Confirm: skip literal Phase 1 (old stack) and merge it into Phase 2 (isolate on
      latest deps directly)? → Yes, confirmed 2026-08-09.
- [ ] Target Python version for the migration?
- [ ] Web stack preference for Phase 3 (e.g. FastAPI + simple frontend vs. something else)?
- [ ] Any Dokploy-specific constraints (Docker required? existing services/ports on the VPS?)

## Working conventions for this project
- Big structural changes (Phase 2 migration) happen on a feature branch, not directly on
  `master`. Merge only after the owner confirms things work.
- This file should be kept up to date as phases complete or decisions get made — treat it
  as the source of truth across sessions, not scratch notes.
