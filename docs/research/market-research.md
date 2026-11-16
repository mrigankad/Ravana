# Ravana market research: face-swap tools, late 2026

Date: 2026-11-16. Method: web search plus fetches of official GitHub READMEs. Evidence quality varies. Primary sources (official READMEs, release notes) are marked **[P]**. Review sites, directories and competitor blogs are marked **[S]** (secondary, often affiliate or SEO content) and are indicative only. Things I could not verify are said so. No numbers below are invented; where figures conflict between sources, both are given.

---

## 1. Tool profiles

### FaceFusion (open source, "industry leading face manipulation platform")
- **Positioning:** Local, privacy-first face manipulation toolkit; the most active open project in the category. Source: [GitHub README](https://raw.githubusercontent.com/facefusion/facefusion/master/README.md) [P].
- **Platform:** Windows, macOS, Linux. README offers a Windows Installer and macOS Installer and says manual install "requires technical skill" and is not recommended for beginners [P].
- **Pricing/licence:** Free. README shows an OpenRAIL-AS badge [P]. One aggregator says commercial use is prohibited and another lists MIT; they conflict, so treat licence terms as "check LICENSE" ([gptzone](https://gptzone.net/aplicaciones-ia/en/facefusion), [SourceForge mirror](https://sourceforge.net/projects/facefusion.mirror/)) [S].
- **Features:** CLI with `run`, `headless-run`, `batch-run`, benchmark, and a job system (create, submit, run, retry, delete) [P]. 3.1.0 added `batch-run` and hififace_256 ([release listing](https://newreleases.io/project/github/facefusion/facefusion/release/3.1.0)) [S]. 3.0 coverage lists pixel boost, lip sync, age modification ([RunDiffusion](https://www.rundiffusion.com/facefusion-3-0/)) [S]. One guide references 3.9.0 (Sept 2026) ([Magic Hour](https://magichour.ai/blog/how-to-use-facefusion)) [S].
- **UX strengths:** Job queue/retry is a real differentiator; installers exist; local processing.
- **Complaints:** CUDA/cuDNN fragility: missing `libcudnn.so.9` leaves app on CPU/slow ([blog](https://zhonguncle.github.io/blogs/21e3b18464ce2171a13a85c5e522b3c9.html)); onnxruntime install silently failing on a Pinokio install ([Pinokio](https://pinokio.co/posts/01kwsxe5jz734pwhwxg86143v1)); RTX 5090 users hitting `libcublasLt.so.12` and onnxruntime-gpu skipped ([r/FaceFusion mirror](https://axebps-redlib.hf.space/r/FaceFusion/comments/1irwn5x/help_installing_cuda_support_for_rtx_5090)); community advice to pin FFmpeg 7.0.2 or earlier because of slow merges in 7.1+ ([r/FaceFusion mirror](https://axebps-redlib.hf.space/r/FaceFusion/hot)) [S]. Creati describes it as needing comfort with local setup ([Creati](https://creati.ai/ai-tools/ai-face-swap/alternatives/ai-face-swap-vs-facefusion-comprehensive-face-swapping-tools-comparison/)) [S].

### Deep-Live-Cam (hacksider)
- **Positioning:** Real-time webcam face swap and one-click video deepfake from a single image. About 98.1k stars, 14.3k forks, AGPL-3.0, v2.1.6 per the README ([GitHub](https://github.com/hacksider/Deep-Live-Cam)) [P].
- **Platform:** Pre-built for Windows, Mac Silicon, CPU, NVIDIA, AMD; manual install on Windows/Linux/macOS, Python 3.11 to 3.14. Providers: CUDA (CUDA Toolkit 12.8.0, cuDNN 8.9.7 for 12.x), CoreML, DirectML, OpenVINO, CPU [P].
- **Pricing:** Free core; "Deep-Live-Cam 2.8 Ultimate" is sold via deeplivecam.net with "30+ exclusive features and priority support" [P].
- **Features:** Mouth Mask, Face Mapping (different faces per subject), webcam via OBS, CLI. Built-in check said to block nudity/graphic content; asks for consent and labelling of deepfakes; says it may add watermarks if legally required [P].
- **Complaints:** Install failures (Apple Silicon pip/torch marker [#1564](https://github-users.yhajoa.workers.dev/hacksider/Deep-Live-Cam/issues/1564), basicsr from source [#266](https://github.lhy31512.workers.dev/hacksider/Deep-Live-Cam/issues/266), startup SyntaxError #608, NoneType shape error #524 via [issue index](https://issues.ecosyste.ms/hosts/GitHub/repositories/hacksider%2FDeep-Live-Cam/issues?page=2)); manual setup requires hand-downloading two ONNX models then pip ([README](https://github.com/hacksider/Deep-Live-Cam)) [P]. CPU-only is impractical for live webcam ([setup guide](https://www.promptzone.com/damonwho/deep-live-cam-setup-tutorial-c0c.md)) [S]. A security-training vendor reports mouth-area defects and FPS fluctuation as the GPU throttled (self-reported, marketing) ([VerSprite](https://versprite.com/security-resources/real-time-deepfake-face-swapping)) [S]. README itself notes insightface models are non-commercial research only [P], a licensing trap for downstream users. Third-party sites flag fake download sources; the README warns to download only from official places [P].

### VisoMaster / VisoMaster-Fusion (ex Rope-next)
- **Positioning:** Pro-style video face swap GUI with fine control. GPL-3.0 ([GitHub](https://github.com/visomaster/VisoMaster)) [P].
- **Platform:** Windows + NVIDIA (installer lets you pick a CUDA version); Linux via Python only; separate requirements for CUDA 11.8 and 12.4. No formal GPU/VRAM minimum stated [P].
- **Features:** Multiple swapper models, DFM model support, per-part masking, XSeg occlusion, expression restorer, LivePortrait-based face editor (expression/pose, RGB tuning of face/hair/brows/lips), live playback before saving, multi-source embeddings, virtual-camera streaming, per-frame video markers, TensorRT [P]. Fusion fork adds auto save/load of workspaces and a swap-input-face-once option ([release](https://newreleases.io/project/github/VisoMasterFusion/VisoMaster-Fusion/release/v1.0.0)) [S].
- **Complaints:** Manual setup is long (Miniconda, Python 3.10.13, conda CUDA, `download_models.py`, then copy extra assets into `dependencies/`) [P]; NVIDIA-only in practice [P]; 102 open issues at fetch time [P]; one Reddit commenter called it "Rope with a facelift" ([thread](https://nyc1.lr.ggtyler.dev/r/StableDiffusion/comments/1ieff29/visomaster_formerly_ropenext_a_new_faceswapping/ma74nc2/?context=3)) [S]. A community job-manager add-on exists ([VisoMaster Job Manager](https://gh.hhj520.top/axel-devs/VisoMaster-Job-Manager)), which signals batch is not built in (inference, not verified).

### Rope (Hillobar) and roop forks (roop-unleashed)
- **Positioning:** Rope: inswapper-based GUI, README calls it an "Ugly GUI, but incredible features and fast workflow", with real-time video player and occlusion functions; Pearl release 2024-05-27 ([GitHub](https://github.com/Hillobar/Rope)) [P-ish via search snippet]. roop-unleashed: roop fork with browser GUI, multi-face selection, many swap modes ([ecosyste.ms](https://awesome.ecosyste.ms/projects/github.com%2Fc0untfloyd%2Froop-unleashed)) [S].
- **Pricing:** Free. **Complaint:** GUI polish (self-described), Windows/NVIDIA focus; Rope largely superseded by VisoMaster. I did not fetch current issue trackers for these two.

### DeepFaceLab (+ DeepFaceLive)
- **Positioning:** The classic train-your-own-model deepfake pipeline. Repo **archived by owner Nov 13, 2024**, read-only ([GitHub](https://github.com/iperov/DeepFaceLab)) [P]. Releases are distributed via magnet link, Mega.nz and Yandex [P]. DeepFaceLive (real-time) is also described as archived with no new model support ([ToolJunction](https://www.tooljunction.io/ai-tools/deepfacelive)) [S].
- **Pricing:** Free. **Complaints:** Unmaintained; per-identity training is slow and expert-oriented (inferred from the pipeline design; I found no sourced complaint quotes).

### Swapface (desktop + credits)
- **Positioning:** Real-time swap for streams/video calls plus image/video. Platform: Windows and Apple-silicon Mac per one review; virtual camera/OBS claims come from directories ([Magic Hour](https://magichour.ai/blog/swapface-review-and-alternatives), [Stork](https://www.stork.ai/en/swapface)) [S].
- **Pricing:** Fragmented. Credits cannot be used for live stream swaps, which need a separate subscription ([ToolWorthy](https://www.toolworthy.ai/tool/swapface)); free plan with watermarks and daily limits ([Toolmage](https://www.toolmage.com/en/tool/swapface/)); "$39/mo" appears in some listings but unconfirmed [S]. Name collides with unrelated sites (swapface.org vs swapfaces.ai).
- **UX:** Product Hunt reviewers mostly praise ease of install and use ([Product Hunt](https://www.producthunt.com/products/swapface-org/reviews)) [S]. **Complaint:** confusing credits vs subscription split.

### Akool
- **Positioning:** Commercial AI video suite (face swap, avatars, translation) with API incl. live face swap. G2 shows 4.8 from 464 reviews per one aggregator ([G2 response](https://www.g2.com/survey_responses/akool-review-9524106), [Capterra](https://www.capterra.com/p/10013539/Akool/reviews/)) [S].
- **Pricing:** Per-seat plus monthly credits; one guide says from $12/mo with API in a $59/mo Pro Max tier ([CheckThat](https://checkthat.ai/brands/akool/pricing)), others say contact vendor [S].
- **Complaints:** Billing: unauthorized charges, trial-to-paid conversion, duplicate billing, slow support (review roundups, [Vmeg](https://www.vmeg.ai/blog/akool-reviews/), [Magic Hour](https://magichour.ai/blog/akool-ai-review)); credit rules unclear [S].

### Reface (mobile)
- **Positioning:** Consumer face swap on videos/GIFs/photos; iOS/Android.
- **Pricing:** A May 2026 market report says a $20/week subscription is being pushed ([MWM](https://mwm.ai/articles/reface-v6-8-0-fixes-critical-bugs-sparking-rating-recovery-in-may-2026)) [S].
- **Complaints:** unexpected recurring weekly charges, confusing cancellation, privacy worries about uploading faces ([Fritz.ai](https://fritz.ai/reface-app-review/)); critical bugs fixed in v6.8.0 with ratings reported at 3.3 to 3.45 depending on aggregator ([MWM](https://mwm.ai/articles/reface-v6-8-0-fixes-critical-bugs-sparking-rating-recovery-in-may-2026)) [S].

### Remaker AI
- **Positioning:** Web/mobile all-in-one; basic photo swap free, multi-face, video and batch swap need credits (30 free on signup) ([Software Testing Help](https://www.softwaretestinghelp.com/remaker-ai-face-swap-review-and-alternatives/)) [S].
- **Pricing:** One-time credit packs that do not expire, per some reviews; "no refunds" per one reviewer ([UniFab](https://unifab.ai/resource/remaker-ai.htm)) [S].
- **Complaints:** outages lasting days, failed generations, "free" tools requiring payment (single review summary, [Fritz.ai](https://fritz.ai/remaker-ai-review/)); lacks fine-tuning; result depends on angle/lighting [S].

### DeepSwap
- **Positioning:** Web face swap for photo/video/GIF. **Pricing:** conflicting: $9.99/mo with 20 credits (Capterra) vs 100 credits (NemoVideo); free tier watermarked ([Capterra](https://www.capterra.com/p/10014580/Deepswap/), [Magic Hour](https://magichour.ai/blog/deepswap-ai-review)) [S].
- **Complaints:** Conflicting retention/refund wording; reviewers advise against confidential footage ([DiyAI](https://diyai.io/ai-tools/video-generation/reviews/deepswap-review/)) [S]. I found no verified complaint threads.

### Picsi.ai
- **Positioning:** Web/Discord face swap on a daily credit system ([AIChief](https://aichief.com/ai-image-tools/picsi-ai/)) [S]. **Pricing:** $9.99/mo in one source, $20-30/mo in another; unconfirmed. **Complaints:** independent feedback is thin; no clear commercial-licence terms found.

### Notable new entrants
I did not find a verified, notable new open-source tool in late 2026. Deep-Live-Cam remains the most discussed ([Menon Lab](https://themenonlab.blog/blog/deep-live-cam-real-time-face-swap-deepfake)) [S]. An April 2026 article about a new free webcam tool is labelled AI-generated with no primary source, so it is excluded ([gentic.news](https://gentic.news/article/open-source-faceswap-tool-enables)). Research is moving to diffusion/video-based swappers (VividFace for flicker, [arXiv](https://arxiv.org/abs/2412.11279v1); LIVINGSWAP, [CVPR 2026 supp.](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Luo_Preserving_Source_Video_CVPR_2026_supplemental.pdf)), but I found no shipping consumer tool built on them.

---

## 2. Comparison table

| Tool | Type | Platform | Price model | Realtime | Batch/queue | Occlusion/edit controls | Install friction | Consent/safety signals |
|---|---|---|---|---|---|---|---|---|
| FaceFusion | OSS | Win/Mac/Linux | Free (OpenRAIL-AS badge) | not confirmed | Yes: jobs, retry | Pixel boost, enhancers (3.0 coverage) | Installers exist, but manual setup is "technical" | Not found in README |
| Deep-Live-Cam | OSS + paid "Ultimate" | Win/Mac/Linux, pre-builts | Free + paid build | Yes (OBS) | CLI only | Mouth mask, face mapping | Pre-builts; manual setup error-prone | Content filter, consent/label notice |
| VisoMaster(-Fusion) | OSS GPL-3 | Win + NVIDIA | Free | Virtual cam | Markers; job manager is a community add-on | XSeg, per-part masks, LivePortrait editor | Installer; manual long | Ethics notice |
| Rope / roop-unleashed | OSS | Win-centric | Free | Rope: live preview | roop-unleashed: multi-face | Occlusion (Rope) | Moderate | Not checked |
| DeepFaceLab | OSS, archived | Win (NVIDIA) | Free | Via DeepFaceLive (archived) | Training pipeline | Full model training | Torrent/Mega zips | Not checked |
| Swapface | Commercial desktop | Win, Mac (Apple silicon) | Credits + separate live subscription | Yes | Unclear | Unclear | Easy per reviewers | Not found |
| Akool | SaaS + API | Web/API | Per-seat credits | Live swap API | API | Limited | None | Likeness rights on the user |
| Reface | Mobile | iOS/Android | Weekly subscription | Camera filters | No | None | None | Privacy concerns |
| Remaker | Web/mobile | Web/iOS/Android | Credits | No | Batch (credits) | "Lacks fine-tuning" | None | Ethics concerns noted |
| DeepSwap | Web | Web | Monthly credits | No | Limited | Limited | None | Unclear retention |
| Picsi.ai | Web/Discord | Web | Daily credits/sub | No | Unclear | Limited | None | NSFW/political barred |
| **Ravana (today)** | OSS MIT SDK + PySide6 GUI | Win (CUDA/DirectML/CPU) | Free | Webcam realtime | Batch + ArcFace scoring | XSeg, 5 restorers, face-select modes | `pip` + exe zip | Optional invisible watermark |

---

## 3. Gaps and pain points across today's apps

1. **GPU/runtime install pain (strongest evidence).** onnxruntime-gpu, CUDA and cuDNN version mismatches are the top issue: FaceFusion cuDNN/cuBLAS missing and silent CPU fallback ([blog](https://zhonguncle.github.io/blogs/21e3b18464ce2171a13a85c5e522b3c9.html), [RTX 5090 thread](https://axebps-redlib.hf.space/r/FaceFusion/comments/1irwn5x/help_installing_cuda_support_for_rtx_5090)); Deep-Live-Cam pins CUDA 12.8.0 + cuDNN 8.9.7 + a specific onnxruntime-gpu and has repeated install issues ([README](https://github.com/hacksider/Deep-Live-Cam), [#1564](https://github-users.yhajoo.workers.dev/hacksider/Deep-Live-Cam/issues/1564)); VisoMaster needs a CUDA-version choice ([README](https://github.com/visomaster/VisoMaster)).
2. **Silent failure and no diagnostics.** The Pinokio report shows a first install failing "without a visible error" and a missing cuDNN leaving the app on CPU ([Pinokio](https://pinokio.co/posts/01kwsxe5jz734pwhwxg86143v1)). No tool I reviewed advertises a "doctor" screen (absence of evidence, not proven).
3. **Non-NVIDIA users are second class.** VisoMaster is NVIDIA-focused [P]; Deep-Live-Cam supports DirectML/OpenVINO/CoreML but offers it via different pinned onnxruntime packages [P]; CPU-only realtime is impractical ([guide](https://www.promptzone.com/damonwho/deep-live-cam-setup-tutorial-c0c.md)).
4. **Manual model downloads.** Deep-Live-Cam manual path: download two ONNX files by hand ([README](https://github.com/hacksider/Deep-Live-Cam)); VisoMaster: run `download_models.py` then copy assets into `dependencies/` ([README](https://github.com/visomaster/VisoMaster)).
5. **Quality artefacts: mouth, hair, occlusion, skin tone, flicker.** Mouth defects in realtime ([VerSprite](https://versprite.com/security-resources/real-time-deepfake-face-swapping)) [S]; hair crossing the face as a source of edge flicker and skin/brightness jumps ([DEV](https://dev.to/genaimedia/why-ai-face-swaps-flicker-in-gifs-5-problems-to-check-54cb)) [S, vendor opinion]; occlusion failures and frame-by-frame flicker in the literature ([FlowFace](https://arxiv.org/pdf/2212.02797), [VividFace](https://arxiv.org/abs/2412.11279v1)). VisoMaster/Rope's heavy manual masking and per-part colour controls exist partly because defaults are not enough [P].
6. **No built-in quality scoring.** None of the READMEs I fetched (FaceFusion, Deep-Live-Cam, VisoMaster) describes identity-similarity or sharpness scoring. Users judge by eye. (Only three READMEs checked; other tools not verified.)
7. **Batch UX is CLI-first or bolted on.** FaceFusion's jobs are CLI commands [P]; VisoMaster's job manager is third-party ([repo](https://gh.hhj520.top/axel-devs/VisoMaster-Job-Manager)); Deep-Live-Cam's batch is CLI [P].
8. **Cluttered or pro-only UIs.** Rope's own README calls its GUI "ugly" [P]; VisoMaster exposes many per-part controls [P]; FaceFusion install is "not recommended for beginners" [P]. Review sites describe a hobbyist/VFX audience ([Creati](https://creati.ai/ai-tools/ai-face-swap/alternatives/ai-face-swap-vs-facefusion-comprehensive-face-swapping-tools-comparison/)) [S].
9. **Consent and provenance are advisory text, not product features.** Deep-Live-Cam: a README disclaimer plus a content filter and a possible future watermark [P]; VisoMaster: an ethics note [P]. Regulatory pressure is rising: EU AI Act Article 50 transparency obligations apply from 2 Aug 2026, with a grace period to 2 Dec 2026 for systems already on the market, layered signed metadata plus imperceptible watermark recommended, deepfakes defined to cover content resembling existing persons ([CEPIC](https://www.cepic.org/post/the-transparency-obligations-of-the-ai-act-take-effect-on-2-august-2026-what-cepic-members-should-consider), [DEV](https://dev.to/mickai/a-watermark-you-can-crop-is-not-provenance-3op1) on watermark weakness) [S]. I did not find face-swap tools shipping C2PA. Ravana's watermark is a head start, but a watermark alone is a label, not provenance.
10. **Non-destructive history and workspace state are rare.** VisoMaster-Fusion's release notes advertise auto save/load of workspaces as a headline add ([release](https://newreleases.io/project/github/VisoMasterFusion/VisoMaster-Fusion/release/v1.0.0)), implying the base lacked it [S, inference]. No evidence of undo/versioned runs elsewhere.
11. **Commercial-side friction (for comparison).** Billing/cancellation complaints dominate: Reface weekly charges ([Fritz](https://fritz.ai/reface-app-review/)), Akool charges and refunds ([Vmeg](https://www.vmeg.ai/blog/akool-reviews/)), Swapface credits vs live subscription split ([ToolWorthy](https://www.toolworthy.ai/tool/swapface)), Remaker "no refunds"/outages. A free, local, no-account app answers this directly, plus privacy of uploads.
12. **Licensing traps.** Deep-Live-Cam credits note insightface models are non-commercial research only [P]; FaceFusion's OpenRAIL-AS has conflicting third-party summaries [S]. Ravana's MIT code still depends on the model licences, which the app should surface.
13. **Dead or fragmented ecosystems.** DeepFaceLab archived Nov 2024 with torrent/Mega distribution [P]; DeepFaceLive archived [S]; fake download sites warned about by Deep-Live-Cam [P].

---

## 4. Opportunities for Ravana (ranked by impact vs effort)

Effort: S under 1 week, M 1-3 weeks, L over a month (my estimates).

| # | Opportunity | Surface | Impact | Effort |
|---|---|---|---|---|
| 1 | **Hardware "Doctor" and first-run wizard.** Detect GPU/driver/CUDA/cuDNN/DirectML, show which provider will be used, and plainly say "running on CPU because cuDNN missing, fix: ...". One-click download of the correct models with progress, checksum and resume. Attacks gaps 1-4. | GUI + CLI (`ravana doctor`) + README | Very high | S-M (extends `first_run_check.py` and the model download CLI) |
| 2 | **Zero-setup Windows installer** (signed exe/MSI with bundled onnxruntime variant picked at install: CUDA, DirectML, CPU). README leads with "Download, double-click". Windows DirectML coverage is a differentiator vs NVIDIA-only VisoMaster. | Release packaging | Very high | M |
| 3 | **Quality score as a first-class UI.** Per-result ArcFace similarity + sharpness chip, pass/warn/fail thresholds, "retry with next preset" suggestion, batch sort by score. No competitor README shows this (gap 6). | GUI + SDK (`result.metrics`) | High | S (metrics exist) |
| 4 | **Non-destructive "runs" history.** Every swap is a saved Run (inputs, config JSON, outputs, score). Compare A/B side by side, re-run with changed settings, undo, export config as reproducible recipe. | GUI + SDK | High | M |
| 5 | **Batch as a visible queue.** Drag a folder, see a table (status, score, time, retry failed, pause/resume, per-item preset), persistent across restarts. Matches FaceFusion's job semantics with a GUI. | GUI | High | M |
| 6 | **Provenance beyond watermark.** Optional signed C2PA-style manifest plus existing invisible watermark; "Export with disclosure" default on; a verify command (`ravana verify out.mp4`). Frame it as helping users meet EU AI Act Article 50 style labelling, without claiming legal compliance. | SDK + GUI + README | High (differentiator, regulatory tailwind) | M-L |
| 7 | **Consent gate.** First-use acknowledgement, optional "consent record" attached to a Run (who, when, source image hash), clear content-policy statement. Turns advisory README text into a product feature. | GUI | Medium-high | S |
| 8 | **Auto quality presets with artefact guards.** Occlusion (XSeg) on by default, mouth-mask-like option, temporal smoothing of mask/landmarks, flicker metric (frame-to-frame delta) reported alongside ID score. | SDK | High | M-L (R&D risk) |
| 9 | **Progressive disclosure UI.** Three modes: Simple (source, target, Go), Tuned (preset + restorer + face select), Expert (all knobs). Counter to the "cluttered/pro" gap. | GUI | Medium | M |
| 10 | **Model licence panel.** Show each model's licence and commercial-use status in-app and in `models list`. Addresses licensing traps. | CLI + GUI | Medium | S |
| 11 | **SDK ergonomics.** `ravana doctor`, typed `Result` with metrics, async/iterator video API, tiny recipes in README, notebook example. | SDK + docs | Medium | S-M |
| 12 | **Virtual camera output** (OBS/Zoom) from webcam mode; VisoMaster and Deep-Live-Cam already offer streaming routes. | GUI | Medium (table stakes for realtime) | M |

Suggested order: 1, 3, 7, 10 (quick wins) then 2, 5, 4, then 6 and 8.

---

## 5. Five UX patterns worth borrowing

1. **Job queue with retry and states** from FaceFusion (`job-create/submit/run/retry`, [README](https://raw.githubusercontent.com/facefusion/facefusion/master/README.md)). Make it a visible table in the GUI.
2. **Live playback/scrub preview before committing a render** from VisoMaster ([README](https://github.com/visomaster/VisoMaster)), plus per-frame markers to override settings on specific frames.
3. **Auto-save and restore workspace** from VisoMaster-Fusion ([release notes](https://newreleases.io/project/github/VisoMasterFusion/VisoMaster-Fusion/release/v1.0.0)). Ravana generalises this into the Runs history.
4. **Pre-built installer picked by hardware** from Deep-Live-Cam (separate builds for Windows, Mac Silicon, CPU, NVIDIA, AMD, [README](https://github.com/hacksider/Deep-Live-Cam)). Ravana can go further by auto-selecting the build in one installer.
5. **Face mapping (different source per detected face)** from Deep-Live-Cam ([README](https://github.com/hacksider/Deep-Live-Cam)); Ravana already has face-select modes, so add a thumbnail strip UI where you click each detected target face and assign a source.

Honourable mentions: Swapface's "live swap in streaming apps" positioning (unverified detail), and consumer apps' single-screen simplicity (Reface/Remaker) for the Simple mode.

---

## 6. Limits of this research
- Pricing for Swapface, Akool, DeepSwap, Picsi came only from review/directory sites that disagree; verify on vendor pages before quoting.
- Reddit/GitHub issue evidence was reached via mirrors and search snippets, not full thread reads. FaceFusion README, DeepFaceLab README and Rope were not fully fetched (connection errors); FaceFusion details beyond the CLI came from secondary sources.
- No verified temporal-flicker complaint threads for a specific open tool; flicker evidence is from papers and practitioner blogs.
- "No competitor does X" statements mean "not found in the sources I read."
