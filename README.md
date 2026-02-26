# foto-destroy

A web app that slowly destroys an image with every single page view. After 1024 views, the image is completely black.

## How it works

Every time someone opens the page, the displayed image is degraded one step further. The destruction is permanent and cumulative — each view applies darkening, glitch effects, and digital corruption that stack on top of previous damage.

Every 10 views, a snapshot is saved so you can scrub through the timeline and watch the destruction unfold.

At view 1024, the image is gone forever (all black). Hit the reset button to start over.

## Effects applied per view

- **Progressive darkening** — image fades toward black with accelerating intensity
- **RGB channel shifting** — red, green, and blue channels drift apart (chromatic aberration)
- **Dead pixels** — random pixels are permanently zeroed out
- **Block corruption** — rectangular chunks are copied to wrong positions (datamosh)
- **JPEG artifacts** — periodic lossy compression creates blocky distortion
- **Pillow filters** — blur, emboss, edge enhance, and median filters degrade structure
- **glitch-this effects** — horizontal displacement, color offset, and scan lines

## Controls

- **Reset button** (top-right) — restores the original pristine image, deletes all snapshots, resets view count to 0
- **Timeline slider** (bottom) — drag to browse through saved snapshots (one every 10 views). Double-click to return to the live view

## Tech stack

- **[Flask](https://flask.palletsprojects.com/)** — Python web framework
- **[Pillow](https://pillow.readthedocs.io/)** — image processing (filters, format conversion)
- **[glitch-this](https://github.com/TotallyNotChase/glitch-this)** — open-source glitch effect library
- **[NumPy](https://numpy.org/)** — float32 image math to avoid 8-bit rounding losses across 1024 cumulative operations

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5001 and start refreshing.

## Deploy

The app includes a `Procfile` for Nixpacks/Coolify deployment. The seed image (`images/default.png`) is bundled in the repo. Runtime state (`source.png`, `original.png`, `snapshots/`, `state.json`) is gitignored.

## State persistence

- `state.json` — stores the current view count
- `images/source.png` — the current degraded image (overwritten every 10 views)
- `images/original.png` — the pristine original (never overwritten, used for reset)
- `images/snapshots/` — one snapshot per 10 views (`snap_0010.png`, `snap_0020.png`, ...)
