# Ravana launch plan: from 6 stars to GitHub trending

Status 2026-11-16. Ravana has 6 stars. The category leader, Deep-Live-Cam, has about 98k. This plan covers what has to be true before launch, the launch sequence, and the product bets that keep stars coming after week one.

## 1. How trending actually works

- GitHub Trending ranks by **stars gained in the window** (daily, weekly, monthly), per language and overall. Total stars barely matter. Roughly 300 to 600 stars in a day puts a Python repo on the daily overall list. About 1,500 or more competes for #1 on a normal day.
- Spikes come from **one viral demo** posted somewhere with reach: X, Reddit, Hacker News, YouTube or TikTok. Deep-Live-Cam's run came from a 10-second webcam clip, not from its README.
- Trending then feeds itself. Newsletters, aggregators and "trending repos" bots pick it up for 2 to 5 days. The README has to convert those visitors.

So the job is: one undeniable 15-second demo, a README and site that convert, and a single coordinated launch day.

## 2. Our hook (why anyone would star this)

The swappers people already use look roughly alike. One line has to explain why Ravana is different:

> **The face swap that tells you how good it is.**

| Hook | Proof we can show | Status |
|---|---|---|
| Every result is scored (ArcFace identity + sharpness) | Score chip on every output, variants ranked by score | Ships (P0) |
| Works on AMD and Intel GPUs, not just NVIDIA | `device="auto"` picks CUDA, then DirectML, then CPU | Ships |
| Installs without CUDA pain | Windows zip plus a Hardware Doctor that explains what's wrong | Zip ships, Doctor is P1 |
| Consent and provenance built in | Consent gate, watermark, signed C2PA-style manifest | Gate and watermark ship, manifest is P2 |
| An SDK, not just an app | `swap_image()` in 3 lines, CLI, batch, evaluate | Ships |
| Polished, light-mode desktop app | Screens in the README and on the site | Ships |

The **score** is the wedge: nobody else ships it and it shows up well in a GIF. The **AMD/Intel** story is the reach: about a third of gaming GPUs are shut out of other tools.

## 3. Launch blockers (must be done before launch day)

| # | Blocker | Owner | Why it matters |
|---|---|---|---|
| B1 | **Real demo media.** A 15 s webcam clip and a 6 s compare-slider GIF with the score updating, recorded on a GPU machine | Maintainer (needs GPU and weights) | No demo, no spike. This laptop has no weights or GPU, so it can't record this |
| B2 | **Release v0.4.0** with `Ravana-0.4.0-windows-x64.zip` attached, built by `packaging/ravana_gui.spec` | CI or maintainer | The README's first instruction must work |
| B3 | **End-to-end smoke test on real weights** for the P0 engine changes (tiling, ROI paste-back, audio remux) | Maintainer | The first issues after launch decide the tone |
| B4 | **GitHub Pages site live** (this repo, `site/`) | Done in this change | The landing page for HN and Product Hunt |
| B5 | Social preview image set in repo settings (1280×640) | Maintainer, by hand. GitHub has no API for it | Every share on X, Slack or Discord shows it |
| B6 | Issue templates, `CONTRIBUTING.md`, 10 `good first issue` labels | Repo | Converts drive-by visitors into contributors |
| B7 | `pip install ravana-sdk` on PyPI | Maintainer | Turns "git+https" into a one-liner |

## 4. Launch sequence

**T−7 to T−1: prepare**
- Record B1. Make three cuts: 15 s vertical (TikTok, Shorts, Reels), 30 s horizontal (X, HN), and a 6 s GIF for the README top.
- Pre-write posts (section 6). Line up 5 to 10 people who will try it on launch day and file honest issues.
- Build the release and check it on a clean Windows VM, with and without NVIDIA.

**T0: launch day (Tuesday or Wednesday, 8:00 US Eastern)**
1. 08:00: Show HN, linking the site, not the repo. The first comment is the maintainer's own "why I built this" note.
2. 08:30: X thread with the 30 s video natively uploaded, the repo link in a reply.
3. 09:00: Reddit: r/StableDiffusion (demo plus settings), r/opensource, r/Python (the SDK angle), r/Amd (the DirectML angle). Each post is written for that community, not cross-posted.
4. All day: answer every comment and issue within 30 minutes. Fix install bugs live and cut 0.4.1 the same day if needed.

**T+1 to T+7: keep the curve up**
- Product Hunt on T+1 (it rewards a second wave).
- Post a "what broke on launch day" changelog. People star repos that visibly move.
- Submit to newsletters and lists: Python Weekly, TLDR AI, Ben's Bites, awesome-face-swap, awesome-python, awesome-onnx.
- Comparison post: "Ravana vs FaceFusion vs Deep-Live-Cam on an RX 7800" with scores, not adjectives.

## 5. Conversion: README, site, metadata

Done in this change:
- **Repo description:** written to lead with the hook and the AMD reach, keeping the search terms people actually type ("face swap", "deepfake", "webcam").
- **Topics:** 20, the GitHub maximum. Ordered for discovery: face-swap, faceswap, deepfake, deep-live-cam alternative terms, onnx, directml, amd, webcam, real-time, desktop-app.
- **Homepage URL:** now the Pages site.
- **README top:** the hook, a "why Ravana" comparison table, a star call to action, and a quick start that fits on one screen.
- **Site** (`site/`, deployed by `.github/workflows/pages.yml`): hero, scores, pipeline, compare, kit, install, responsible use. Light, Inter, the same palette as the app.

Still needed: swap the hero still for the B1 GIF once it exists (`site/index.html` `#demo`, README line 2).

## 6. Draft posts

**Show HN:** "Show HN: Ravana, an open-source face swap that scores its own results". Body: three sentences on the score, AMD support and consent gate, a link to the site, and the honest limitations (the model licences are non-commercial).

**X:** "Every face swapper makes you judge by eye. Ravana scores every result: identity 0.71, sharpness +16.9. Runs on NVIDIA, AMD and Intel. MIT, local, no cloud. [video]"

**r/Amd:** "I got real-time face swap running on DirectML. No CUDA needed. Open source." Lead with fps on a named card.

## 7. Product bets that keep stars coming (post-launch roadmap)

Ranked by how much new reach each bet adds, against effort.

| Rank | Bet | Why it spreads | Effort |
|---|---|---|---|
| 1 | **Virtual camera** (OBS, Zoom, Meet) | The live webcam clip is the most shared format in the category | M |
| 2 | **Hardware Doctor** | It fixes the #1 complaint across competitors, CUDA install pain | S |
| 3 | **Score-ranked auto mode:** try 4 settings, keep the best | "It picks the best result for you" makes an easy demo | S |
| 4 | **TensorRT FP16 + IOBinding** | A published fps number starts benchmark threads | M |
| 5 | **macOS (CoreML) build** | Opens a new audience and new subreddits | M |
| 6 | **Signed provenance manifest (C2PA)** | Press angle: "the responsible deepfake tool" under the EU AI Act | M |
| 7 | **ComfyUI node + Python API docs site** | Taps the ComfyUI audience and its custom-node lists | S |
| 8 | **Hugging Face Space demo** | Try it in a browser with no install, which converts sceptics | S |
| 9 | Diffusion-based video swap (VividFace-style) for stills | Research credibility | L |

Engineering detail for 2, 4 and the GPU path is in `engineering-audit.md` sections D to F.

## 8. Risks

- **Platform policy.** GitHub's Acceptable Use Policies ban projects built for non-consensual sexual imagery and fraud. Ravana's consent gate, labelling guidance and "use it responsibly" section must stay prominent. Never market it with celebrity or politician swaps; demos use the maintainer or consenting volunteers.
- **Model licences.** The defaults (InsightFace, HyperSwap) are non-commercial. Say so up front. Hiding it would become the top HN comment.
- **Install bugs on launch day** kill momentum faster than anything. B2 and B3 are not optional.
- **Fake download sites** appear once a repo trends (Deep-Live-Cam had them). Say in the README that Releases is the only official download.

## 9. Metrics

| Metric | Target, launch week | Target, 90 days |
|---|---|---|
| Stars | 1,500 (aim for daily top 10) | 8,000 |
| Release downloads | 3,000 | 25,000 |
| Install success (from Doctor telemetry, opt-in only) | above 90 % | above 95 % |
| Issues answered within 24 h | 100 % | 90 % |
| External contributors | 5 | 30 |
