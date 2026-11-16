# Ravana character + icon kit

Rebuild everything with:

```bash
.venv/Scripts/python design/kit/build_kit.py
```

The script writes:
- `design/kit/icons/*.svg` and `design/kit/character/*.png`
- the contact sheets `icons-sheet.png` and `character-sheet.png`
- copies into `demos/gui_app/assets/{icons,character}` and `docs/assets/brand/`

## Icon style

| Rule | Value | Why |
|---|---|---|
| Grid | 24 × 24, 2 px live-area padding | Same metrics as Lucide/Feather, so it sits next to Inter at 13 to 15 px |
| Stroke | 1.75, round caps and joins | Matches Inter's stroke weight at UI sizes. 2.0 looked heavy next to 13 px text |
| Fill | None, except small dots and the red record dot | Line icons keep the white UI quiet, so the photos stay the hero |
| Colour | Ink `#15120F` by default, tinted at render time (muted `#6B645B` for secondary) | One source SVG per icon |
| Corners | 2 px radius on rectangles | Echoes the 8 to 10 px radii of the UI at 24 px scale |
| Metaphors | Literal over clever: a hand for consent, a shield for watermark, a rosette for provenance | People meet these icons at moments of trust and should not have to decode them |

Research basis: Lucide, Phosphor and Tabler all converge on a 24 px grid, a 1.5 to 2 stroke and round joins for product UI. Desktop face-swap tools (FaceFusion, VisoMaster, Deep-Live-Cam) mostly ship text buttons or mixed icon packs, so a single coherent set is part of the "polished" gap in `docs/research/market-research.md`.

## Where every icon is used

| Group | Icons | Surface |
|---|---|---|
| Navigation | swap, batch, live, models, history*, doctor*, settings* | Title bar tabs (*P1 tabs) |
| Inputs | source-face, target-image, target-video, drop, folder, folder-open, replace | Drop cards, file dialogs, replace-input action |
| Settings | quality, restore, pixel-boost, faces, faces-multi, gpu, cpu, mask, lighting, temporal, sliders | Rail rows, device chip, docs feature table |
| Stage | compare, side-by-side, result, zoom-in, zoom-out, fit, save, reveal, copy, variants | View switcher, canvas toolbar, Save / Show in folder |
| Status & metrics | score, identity, sharpness, timer, check, warning, error, info, spinner, no-face | Readout captions, banners, batch row states |
| Batch | add-file, add-folder, trash, retry, play, pause, stop, queue, sort | Queue toolbar and rows |
| Live | camera, camera-off, record, virtual-cam, fps | Camera tab, live HUD |
| Models | download, verified, licence, hash, storage | Models tab: download, checksum, licence panel, disk use |
| Trust | consent, watermark, provenance, local-only, eye-off | Consent gate, watermark toggle, manifest viewer, "nothing leaves this machine" |
| Utility | chevron-down, chevron-right, close, more, keyboard, external, terminal, docs, code | Selects, menus, shortcuts sheet, README and docs links |

## Dasha: the character

Dasha (from *Dashanana*, "ten-faced") is the little king. He has ten heads, so he has ten states, and each pose is tied to one app state. Use him only where he helps someone understand a state. He never appears next to a result image, because the result is the hero.

| Pose | App state | Where it shows |
|---|---|---|
| idle | Ready, nothing loaded | Empty canvas on the Swap tab |
| working | Long job running (video) | Canvas while a video swap runs |
| success | Strong identity score, batch finished | OK banners |
| no-face | Detector found no face | Canvas plus error banner after a no-face failure |
| error | Any other failure | Error banners |
| consent | First-run consent gate | Consent banner |
| offline | Models missing | Models tab canvas, download banner |
| batch | Queue empty or ready | Batch tab canvas |
| live | Camera tab, not running | Live tab canvas |
| sleeping | Long idle, camera stopped | Reserved (screensaver / idle after 10 min) |

**Rules**
- **Sizes:** 40 px in banners, 96 to 168 px on the canvas (it scales with height), never above 200 px.
- **One at a time:** never show two poses on screen together.
- **Copy carries the meaning:** the pose supports the text. Every state also works for someone who can't see the image.
- **Source art:** `design/character/dasha-expressions.png` (Paper file "Ravana — Redesign", page 00). These cuts come from a 1x export, so re-export the expression sheet from Paper at 2x and rerun the build for sharper poses.
