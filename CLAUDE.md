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
Phases 1–3 (dependency modernization, web app) are merged into `master` and deployed.
The gallery-sync and frontend-redesign work below (originally two separate branches)
is combined onto this one branch, open as a single PR against `master`.

- **Stack**: Python 3.14.2, torch 2.13.0 / torchvision 0.28.0, FastAPI + vanilla JS
  for the web app.
- **Live app**: https://bts.subhajitdas.me — deployed via Dokploy (project "Brain
  Tumor Segmentation", app "Web"), auto-deploys on push to `master`.
- History: modernization work happened on `modernize-deps` (PR #28), the web app on
  `web-app` stacked on top (PR #29). Both merged into `master` 2026-08-09.

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
  - `images/` at repo root is **not** the gallery's source of truth at runtime —
    `.dockerignore` excludes it entirely, so the deployed app never reads from it.
    It only feeds two build-time/doc consumers: (1) `web/scripts/build_gallery_thumbs.py`,
    a one-off script that globs `images/*.png` to regenerate the committed
    `web/static/gallery/{thumbs,full}/*.webp` (295 of each, already committed —
    these are what `/gallery` actually serves), and (2) `README.md`'s "Results"
    section, which hand-picks 20 of the highest-scoring images by exact filename.
    As of 2026-08-10, `images/*.png` is trimmed in git to just those 20
    README-referenced files (whitelisted in `.gitignore` by exact filename, not a
    dice-score prefix range) to cut repo bloat. This means `build_gallery_thumbs.py`
    can no longer regenerate the full 295-image set from what's in git — the source
    corpus (601 images, synced 2026-08-10 from the owner's Google Drive zip) was
    never committed at all; re-derive from that zip if the gallery ever needs
    rebuilding at a different size/quality/threshold.
  - `bts/classifier.py` and `bts/model.py` lazily import `tensorboard`/`torchinfo`
    (scoped to `train()`/`.summary()`) rather than at module level — needed so the
    lean web image (which deliberately excludes those training-only deps) can still
    `import bts.classifier`/`bts.model` for inference.
  - Dockerfile installs CPU-only torch/torchvision via a scoped
    `--index-url https://download.pytorch.org/whl/cpu` (avoids the much larger
    default CUDA-bundled wheels). Final image: ~1.06GB.
  - **In-house per-IP rate limiter** (`web/rate_limit.py`) gates `/api/predict`.
    Skippable in dev via `APP_ENV=development`; unset or any other value (including
    the Dokploy deploy, which sets nothing) rate-limits — the safe default.
- **Dokploy deployment**: uses the existing GitHub App integration on that instance
  (no new auth needed), builds directly from the repo's root `Dockerfile`. No
  `serverId` set — deploys on the Dokploy host itself (no separate remote servers are
  registered on this instance). Domain `bts.subhajitdas.me` → port 8000, https,
  Let's Encrypt.

## Frontend redesign
Full visual redesign of the 3 pages (`/`, `/about`, `/gallery`) plus a new 404 page,
requested by the owner. Not yet reviewed/merged.

- **Styling**: Tailwind CSS via the `cdn.tailwindcss.com` play-CDN script (owner's
  explicit choice — no build step). It logs a "should not be used in production"
  console warning; known and accepted, not a bug. `web/static/css/style.css` now only
  holds what the CDN build can't express: `@font-face`-adjacent Google Fonts links,
  the corner-bracket `.viewport` component, dialog skin, focus rings, and
  `prefers-reduced-motion` handling.
- **Visual identity**: a radiology-reading-room / PACS-viewer aesthetic — near-black
  background, off-white "film" body text, a mono "readout strip" under the header
  showing real model stats (`DynamicUNet`, `512×512×1`, mean Dice `0.74`). Fonts:
  Space Grotesk (display), IBM Plex Sans (body), IBM Plex Mono (data/labels/nav).
  The one accent color (`#e8543c`, "finding") is deliberately the same red used by
  `web/inference.py`'s `_make_overlay()` tumor tint — the UI's accent literally is the
  product's own output color, not an arbitrary pick.
  - **Signature device**: a recurring corner-bracket "viewport" frame
    (`.viewport` in `style.css`) wraps every piece of inspectable imagery site-wide
    (sample thumbs, result images, gallery thumbs, about-page diagrams) — brackets
    turn red on hover/selection. The 404 page reuses the same empty-viewport
    vocabulary ("No series loaded") for both the missing-page state and a failed
    prediction's mask/overlay panes.
  - Loading state for the predicted-mask/overlay panels: don't rely on the `hidden`
    attribute on an element that also carries a Tailwind `flex` class — `[hidden]`
    and `.flex` are equal-specificity, and whichever Tailwind emits later in its
    generated stylesheet wins, so `el.hidden = true` can silently no-op. Use
    `el.style.display` (inline styles always win) instead — see `setLoading()` in
    `web/static/js/main.js`.
- **Social card**: `web/static/social/og-card.png` (1200×630) and
  `web/static/favicon.svg` / `favicon-32.png` / `apple-touch-icon.png` are generated
  images, not hand-drawn — a one-off Pillow script (not committed; lived in the
  session scratchpad) rendered them using the *actual* Space Grotesk / IBM Plex Mono
  webfont files (downloaded from the `google/fonts` GitHub repo, since neither font
  is installed locally) so the social card's typography matches the live site
  pixel-for-pixel rather than approximating it with a system font. Re-derive that
  script if the OG card ever needs updating — the source `.py` wasn't kept.

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
