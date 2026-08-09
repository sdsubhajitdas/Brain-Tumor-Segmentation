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
- **Training path exercised for real (owner asked to train a tiny model, not just
  test inference) — found and fixed 3 more torch-version incompatibilities in
  `bts/classifier.py`**, none of which were caught by the earlier inference-only
  verification since `train()`/`test()` were never actually called until now:
  - `optim.lr_scheduler.ReduceLROnPlateau(..., verbose=True)` — `verbose` kwarg was
    removed in current torch with no replacement; dropped it (only loses the
    automatic "reducing learning rate" print, no behavior change).
  - `BrainTumorClassifier.test()` used `testloader.next()` (Python 2 iterator style)
    — current torch's DataLoader iterator no longer exposes `.next()`; changed to
    `next(testloader)`.
  - `BrainTumorClassifier.test()` also never converted `mask` to a numpy array
    (unlike `predict()`, which does). On the old stack `np.multiply(ndarray, tensor)`
    apparently didn't matter; on the current numpy/torch interop it returns a
    `torch.Tensor`, and `Tensor.sum()` doesn't accept numpy's `axis=` kwarg, crashing
    `_dice_coefficient`. Fixed by adding `.numpy()` at the same point `predict()`
    already does it.
  - Verified via a temp scratch-only script (not committed, not part of the repo):
    trained a **fresh** `DynamicUNet` (not the saved checkpoint) on 32 real samples
    for 5 epochs, ran `BrainTumorClassifier.test()` and `.predict()` against it, then
    ran the real unmodified `api.py` `Api` class (subclassed only to point
    `_load_model` at the scratch checkpoint) end-to-end: train → test → predict → CLI
    inference all working. `saved_models/UNet-[16, 32, 64, 128, 256].pt` confirmed
    byte-identical (checksum) before/after — never loaded or overwritten.
  - `bts/loss.py` (`DiceLoss`/`BCEDiceLoss`) and `bts/plot.py` (`loss_graph`,
    `result`) both exercised for real as part of this and needed no changes.

### Phase 3 — Web app to host the model
**Status: v1 implemented, verified locally (including in Docker), not yet deployed to
the actual Dokploy VPS. Branch `web-app`, stacked as PR #29 on top of #28
(`modernize-deps`, still unmerged) — `--base modernize-deps`, not `master`.**

- Owner confirmed requirements (2026-08-10): visitors upload an image OR pick from
  ~10-15 curated samples, run the model, see the predicted mask; a page section
  adapting README content (architecture, augmentation, dataset, training) — not raw
  markdown, README.md untouched; a results gallery, thumbnailed/lazy-loaded for VPS
  egress cost. **New constraint given same session: the entire backend+frontend must
  run inside a single Docker image for Dokploy.**
- Stack confirmed: FastAPI + plain server-rendered HTML/vanilla JS (no SPA framework).
  Multi-page (`/`, `/about`, `/gallery`), not a JS-tabbed single page.
- Full implementation plan written and approved — see
  `/Users/subhajitdas/.claude/plans/eager-nibbling-panda.md` for the complete design
  rationale (routes, image pipeline, Dockerfile strategy, verification plan).
- Built: `web/` package (FastAPI app, `inference.py` wrapping `BrainTumorClassifier`
  for load-once-reuse, routes, Jinja2 templates, vanilla JS), `Dockerfile`,
  `.dockerignore`, `requirements-web.txt` (deliberately separate from
  `requirements.txt` — excludes training/notebook-only deps like `jupyterlab`).
- 12 curated sample pairs and 88 gallery images (WebP thumbs + capped full-size) are
  pre-baked and committed under `web/static/` by one-off scripts
  (`web/scripts/prepare_samples.py`, `web/scripts/build_gallery_thumbs.py`) — the web
  app never depends on the large gitignored `dataset/` folder, and thumbnails aren't
  regenerated on every deploy.
- **Found and fixed 2 more real bugs** in `bts/` while getting this to actually run in
  Docker: `bts/classifier.py` and `bts/model.py` imported `tensorboard`/`torchinfo` at
  module level (for `train()`'s `SummaryWriter` and `.summary()`), which broke
  `import bts.classifier`/`bts.model` entirely inside the web image since neither
  package is installed there (deliberately excluded, training-only). Fixed by making
  both imports lazy, scoped to the methods that use them — no behavior change.
- **Verified end-to-end, not just "it runs":** local smoke test of every route; a
  pixel-perfect cross-check of `/api/predict`'s output against the already-verified
  `api.py` CLI (`numpy.array_equal`, zero differences) on **both** `mps` (host) and
  `cpu` (Docker container); Docker build+run (final image **1.06GB**, CPU-only
  torch/torchvision via a scoped `--index-url` install to avoid the default
  CUDA-bundled wheels); upload edge cases (non-image → 400, oversized → 413, a real
  non-MRI photo → succeeds non-crashing); confirmed the in-house per-IP rate limiter
  triggers (429) correctly; confirmed `/healthz` stays responsive (28ms) while
  concurrent inference requests are in flight, verifying the threadpool-offload
  actually prevents event-loop blocking.
- Uploaded images are processed entirely in memory and never written to disk — no
  cleanup job needed (owner asked about this explicitly during planning).
- Not yet done: actual deployment to the owner's Dokploy VPS (no VPS access this
  session — port/domain/health-check-path config happens in the Dokploy UI); syncing
  the larger Google Drive results corpus into the gallery (size/count still unknown,
  v1 gallery uses the 88 images already in the repo).

## Open questions to resolve with the owner
- [x] Confirm: skip literal Phase 1 (old stack) and merge it into Phase 2 (isolate on
      latest deps directly)? → Yes, confirmed 2026-08-09.
- [x] Target Python version for the migration? → Latest (confirmed 2026-08-10). Already
      what's in use: Python 3.14.2, the system Python on this machine. No action needed.
- [x] Web stack preference for Phase 3? → FastAPI + plain HTML/vanilla-JS, confirmed by
      owner 2026-08-10, implemented same session (see Phase 3 above, PR #29).
- [ ] Any Dokploy-specific constraints (Docker required? existing services/ports on the VPS?)
      Owner will share this later (2026-08-10) — do not assume Dokploy specifics until then.
      This is now the actual blocker on deploying PR #29 for real.

## Working conventions for this project
- Big structural changes (Phase 2 migration) happen on a feature branch, not directly on
  `master`. Merge only after the owner confirms things work.
- This file should be kept up to date as phases complete or decisions get made — treat it
  as the source of truth across sessions, not scratch notes.
