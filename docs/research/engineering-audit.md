# Ravana engineering audit (as of 2026-11-16, v0.3.4)

Scope: read-only audit of `face_swap/**`, `demos/gui_app/**`, tests, CI, packaging. No repo files were modified. Line numbers refer to the working tree at audit time. Speed numbers in section D are labelled estimates derived from reasoning, not measurements.

---

## A. As-is architecture

### A.0 Two pipelines, one is dead for default presets
`FaceSwapPipeline` has a **native path** (`_process_frame_native`, pipeline.py:565) used for all shipped presets (InSwapper/HyperSwap are both "native", `_native_inswapper_enabled` pipeline.py:345) and a **modular path** (`_process_frame_modular`, pipeline.py:932: RetinaFaceDetector -> InsightFaceLandmarkDetector -> FaceAligner -> swapper.swap -> FaceBlender) used only for SimSwap/plugins. `initialize()` still instantiates modular components (pipeline.py:196-206) in the native case.

### A.1 Stage table (native path)

| # | Stage | Module / file:line | Model | Input -> output | Device |
|---|---|---|---|---|---|
| 1 | Input decode | `cv2.imread` (api.py:195); `VideoCapture.read` (api.py:286); webcam `cap.read` (api.py:386, workers.py:226) | none | file -> HxWx3 BGR uint8 | CPU (software decode) |
| 2 | Detect (+ landmarks + embed bundled) | `FaceAnalysis.get` via `_detect_faces` pipeline.py:357; adaptive ladder `detect_faces_adaptive` adaptive.py:137; fast path downscale + det_size 320 (fast_video.py:130) | InsightFace `buffalo_l`: `det_10g` SCRFD (640x640 padded, NCHW f32), `w600k_r50` ArcFace (1x3x112x112 -> 512), plus `1k3d68`, `2d106det`, `genderage` when not fast (pipeline.py:174-176) | BGR frame -> list of `Face` (bbox, kps 5x2, det_score, 68/106 pts, 512-d embedding) | ORT CUDA / DML / CPU; blob prep and SCRFD decode/NMS are numpy on CPU |
| 3 | Source embed (once) | `extract_source_embedding` pipeline.py:1017 | ArcFace; `normed_embedding` cached as `_source_face` | image -> 512-d | one-off; may detect up to 3x (pipeline.py:1029, 1040-1047) |
| 4 | Select | `select_faces` adaptive.py:323; `_get_faces_video` pipeline.py:416 (detect every N + ROI re-detect) | none | faces -> subset | CPU |
| 5 | Align (5-pt similarity crop) | HyperSwap `face_align.norm_crop2(frame, kps, 256)` hyperswap.py:227; InSwapper inside `INSwapper.get` (128) | none | frame -> 256x256 (or 128) BGR + 2x3 M | CPU `warpAffine` |
| 6 | Swap | `HyperSwapModel.swap_face` hyperswap.py:212; `InSwapperModel.swap_face` inswapper.py:105 | HyperSwap-1A 256 (`source` 1x512 f32, `target` 1x3x256x256 f32 in [-1,1]) or inswapper_128 (target 1x3x128x128 [0,1], latent 1x512 x emap) | -> 1x3xSxS | ORT; numpy pre/post (BGR->RGB, /255, transpose, f32) on CPU; `session.run` copies H2D and D2H each call |
| 7 | Paste-back / blend | `_paste_back_insightface` hyperswap.py:34; InSwapper uses insightface's | none | fake crop + M + full frame -> **full-frame** composite | CPU, full-frame float32 |
| 8 | Occlusion | `XSegOccluder.mask_for_bbox` xseg.py:202, `apply_occlusion_blend` xseg.py:42; hull masks adaptive.py:842 | XSeg `xseg_1` (256x256) | padded bbox crop of **original** frame -> soft mask | ORT; resize/blur CPU |
| 9 | Colour / lighting | `_color_match_face_region` pipeline.py:739 -> adaptive.py:526/569/703 | none (Reinhard LAB, YCrCb skin mask, low-freq luminance transfer) | crop pair -> crop | CPU float32 |
| 10 | Lower-face preserve | `_maybe_preserve_lower_face` pipeline.py:859 (+ luminance beard heuristic adaptive.py:672) | none | crop -> crop | CPU |
| 11 | Restore + "pixel boost" | `enhance_face_region` enhancer.py:677 -> `enhance_with_tiles` enhancer.py:626 -> GFPGAN 1.4 / GPEN-512 / CodeFormer / RestoreFormer++ ONNX | 512x512 NCHW f32 [-1,1] | padded **bbox** crop (not aligned) upscaled Lanczos to target_px, tiled 512 | ORT; resize/tiling CPU |
| 12 | Post-restore re-match | chroma-only colour, grain match (pipeline.py:668-675) | none | | CPU |
| 13 | Temporal | `ema_smooth_insight_faces` smoother.py:321 (bbox/kps EMA, IoU match) before swap; `OpticalFlowSmoother.smooth_frame` optical_flow.py:71 (Farneback on full frame, warps previous output, blends 0.28) after swap | none | | CPU cv2 |
| 14 | Watermark (optional, default off) | `InvisibleWatermarker.embed` watermarker.py:117 (DCT, blue channel) | none | | CPU |
| 15 | Tiny-face downscale | `cv2.resize(INTER_AREA)` pipeline.py:680 | | | CPU |
| 16 | Encode | `cv2.VideoWriter('mp4v')` api.py:280; `cv2.imwrite` | | BGR -> mp4v | CPU. **No audio mux** |

### A.2 Per-path flow
- **Image** (`swap_image`, api.py:171): builds a *new* `FaceSwapPipeline` (api.py:208) -> `initialize()` loads buffalo_l (5 ONNX), HyperSwap (~400 MB), GFPGAN (~340 MB), XSeg -> source embed -> `process_frame`. Per face, serial: swap -> xseg -> colour -> preserve -> (optional ID re-detect); then per face: enhance -> chroma match -> grain -> preserve.
- **Video** (`swap_video`, api.py:219): single-threaded `read -> process_video_frame -> write`. `process_video_frame` adds `_get_faces_video` (detect every N, ROI re-detect between; pipeline.py:416-457), face EMA, native path, then flow smoother. `seamless` sets detect_every_n=2.
- **Webcam** (`start_realtime_swap` api.py:307, `demos/webcam_demo.py`, GUI `WebcamWorker` workers.py:173): same single-threaded loop with `FastVideoConfig` (det_size 320, detect downscale 640, skip enhance, XSeg off, max_faces=1, ROI track). `async_detection` / `AsyncFaceDetector` is consulted **only in the modular path** (pipeline.py:940), so it is a no-op for the default path.

### A.3 Threading / queueing
Effectively none. The pipeline is synchronous and single-threaded; only the unused-by-default `AsyncFaceDetector` thread (async_detector.py:124) and Qt `QThread` workers exist. No frame queue, no ring buffer, no stage overlap, no CUDA streams, no batching (`PipelineConfig.batch_size` pipeline.py:93 is never read). One ORT session per model, each with its own arena.

---

## B. Gaps and defects

### B1. Performance
1. **Pipeline re-created per call.** `swap_image` (api.py:208) and `swap_video` (api.py:263) build and initialise all sessions every call; `BatchWorker` builds an extra pipeline per video (workers.py:296). GUI re-runs reload ~1 GB of weights. No session or pipeline cache.
2. **Full-frame paste-back per face.** `_paste_back_insightface` warps three full-frame images, runs `np.where` over the whole frame (hyperswap.py:60), erodes/blurs full-frame masks, blends in float32 (hyperswap.py:49-81). Cost scales with frame area, not face area.
3. **Dead compute.** `fake_diff` is computed, warped, dilated and blurred (hyperswap.py:41-76) but never used in the merge (hyperswap.py:78-81).
4. **Tile schedule bug.** `_tile_starts(1024, 512, 64)` returns `[0, 448, 512]` (enhancer.py:600-609), so a 1024 boost runs **9** GFPGAN passes per face (4 would cover it). README shows ~2.3 s/image for GFPGAN at boost 512 on DirectML.
5. **Many full-frame copies per face.** `work.copy()` (pipeline.py:596), `swapped.copy()` in colour match (818), grain (856), preserve (912), `apply_occlusion_blend` (xseg.py:80), `enhance_face_region` (enhancer.py:756). Each stage copies the full frame to write back a crop.
6. **Full-frame masks.** `landmark_hull_mask((h, w), ...)` allocates and blurs a full-frame mask per face (pipeline.py:733, 782; adaptive.py:854) then slices a crop.
7. **No IOBinding, CUDA graphs or provider options anywhere** (grep clean). Providers are bare strings (providers.py:42-83), default arenas across many sessions (VRAM bloat), numpy in/out with synchronous H2D and D2H per call. TensorRT code (`optimization/runtime.py`, `export.py`) is **not imported by the pipeline** and uses the legacy binding API (`num_bindings`, `get_binding_name`, removed in TensorRT 10) plus pycuda.
8. **DirectML options.** ORT docs require `enable_mem_pattern=False` and `ORT_SEQUENTIAL` for DML ([ORT DML docs](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)); HyperSwap/GFPGAN/XSeg/CodeFormer use defaults (hyperswap.py:115, enhancer.py:143, xseg.py:114), and FaceAnalysis sessions are created by insightface with defaults. DML also permits one thread in `Run` per session.
9. **Whole bundle runs per detect.** Non-fast mode runs `1k3d68`, `2d106det`, `genderage` and ArcFace for every face (pipeline.py:174-176). Target embedding is only needed for `swap_weight != 0.5` or `pose` select.
10. **Detection retry ladder.** `detect_faces_adaptive` can run ~7 full `FaceAnalysis.get` calls on a face-less frame and calls `prepare()` on size changes (adaptive.py:151-200); multiplier on video frames with no face.
11. **Tiny-face path upscales the whole frame** up to 3-4x cubic, re-detects, and pastes back at that resolution (pipeline.py:583-592, adaptive.py:373-426). A 1080p frame x3 is ~5760x3240. Should crop an ROI.
12. **CPU optical flow on the full frame** (Farneback, optical_flow.py:171) plus `np.mgrid` allocation per frame (optical_flow.py:259). The RAFT option builds `raft_small(weights=...)` on **every call** (optical_flow.py:219).
13. **ID check = full re-detect per face** (`_check_id` pipeline.py:915-930), off by default.
14. **Python loops.** `lower_face_preserve_mask` row loop (adaptive.py:657); watermark per-8x8 block `cv2.dct` loops, and extract over every block (watermarker.py:157-171, 186-198).
15. **SHA-256 of ~400 MB files at every load** (model_manager.py `ensure_downloaded`/`_verify_sha256`). Cache a stamp keyed by (size, mtime).
16. **GUI frame handling.** `WebcamWorker` emits `out.copy()` every frame through a queued signal with no drop policy (workers.py:231); video preview decodes on the GUI thread (preview.py:267-284).
17. **Profiler is coarse.** Only detection/swap/blend/watermark (pipeline.py:574-678); colour and xseg are folded into "swap", restore into "blend"; the flow smoother is outside stages.

### B2. Correctness / quality risks
1. **Restorer fed unaligned bbox crops.** GFPGAN/GPEN/CodeFormer/RestoreFormer++ expect FFHQ-aligned 512 faces; Ravana resizes the padded bbox (enhancer.py:171, 378, 532) and for boost 1024 **tiles** it into 512 patches (enhancer.py:644-674). Face-prior models hallucinate per tile. FaceFusion applies "pixel boost" to the *swapper*, not the restorer, so this is a mislabel and a plausible cause of the weak ID score (README: ID 0.309 GFPGAN vs 0.324 OpenCV).
2. **Global temporal ghosting.** `OpticalFlowSmoother` warps the previous *whole output* and blends 28% into the whole frame, then stores the blend as the next "previous" (optical_flow.py:109-126, pipeline.py:547-551): a recursive IIR on the full frame; background smears and the face lags. The `_flow_accum` EMA of per-pair flow fields (optical_flow.py:97-102) is not a valid flow accumulation.
3. **EMA lag and stale caches.** bbox/kps EMA alpha 0.35 (smoother.py:321-379) lags fast motion; on non-detect frames with no ROI result, cached faces are reused frozen (pipeline.py:457), so the pasted face trails the head. One-Euro is the standard fix.
4. **Identity flips with several faces.** Faces are sorted by `det_score` per frame (pipeline.py:400) then `first`/`index` applied (adaptive.py:348-355). No tracker, so the swapped face can change frame to frame.
5. **ROI-only re-detect misses newcomers** until the next full detect (pipeline.py:439-455).
6. **XSeg gets an unaligned crop** of the original frame (xseg.py:175-194, pipeline.py:613); FaceFusion uses the aligned 256 crop. Blend is bbox-rectangular, not the paste region.
7. **Skin-model bias.** Fixed YCrCb range (adaptive.py:516-518) and a luminance beard heuristic (adaptive.py:699-700) are biased against dark skin, tinted light and shadowed jaws; colour stats are fitted on masks derived from them.
8. **RGB/BGR bug.** `RetinaFaceDetector` converts BGR->RGB before `FaceAnalysis.get` (retinaface.py:87-94, 130-135) but insightface expects BGR. It also hard-codes CUDA/CPU providers (retinaface.py:58-62), ignoring DML/auto.
9. **Watermark defeated by own pipeline.** Embedded at pipeline.py:677, then the tiny-face path downsizes the frame (pipeline.py:680-686). Sign-of-coefficient detection with fixed delta and payload in the first rows of blocks (watermarker.py:151-171) will not survive H.264, resize or crop despite the docstring (watermarker.py:9-10). Default OFF (pipeline.py:276), at odds with PRD section 6.3 intent.
10. **Audio is never muxed.** README says audio is remuxed when ffmpeg is available, but `swap_video` writes `mp4v` via `cv2.VideoWriter` (api.py:280) and `demos/cli.py` has no ffmpeg/audio calls. `AudioProcessor` is only used in examples/tests. GUI video output is silent. `mp4v` is also large/low quality, and `.mov`/`.mkv` outputs reuse it.
11. **Exceptions as control flow.** HyperSwap retries `run` up to three times with other input orderings on any exception (hyperswap.py:236-245), masking OOM/provider faults.
12. **Mutating shared Face objects.** `offset_faces`, `scale_faces_from_detect_space`, `_shift_face`, `upscale_if_small_face` rewrite `bbox`/`kps` in place (fast_video.py:100-109, 123-125; adaptive.py:123-134, 458-460); cached faces can be shifted twice.
13. **IO.** `cv2.imread` fails on non-ASCII Windows paths (api.py:195-202, workers.py:134-135); `fps==0` and `VideoWriter.isOpened()` unguarded (api.py:274-281); error messages interpolate the variable after it was overwritten with `None` (api.py:196-197, 201-202).
14. **Registry and downloads.** `ModelRegistry` sorts versions lexicographically (model_manager.py:76, "0.10" < "0.9"); SHA mismatch after download only warns; InSwapper has no checksum (inswapper.py:94-99); all paths are `./models` relative to CWD (pipeline.py:174, 209).
15. **Platform stubs.** `apple.py` sets `has_ane = is_arm` inside a `try: pass` (never raises); `native/` C++ and `platform/mobile.py` are not wired into the SDK path.

### B3. GUI (demos/gui_app)
1. **Signal shadowing.** `SwapWorker`, `ScoreWorker`, `BatchWorker`, `ModelsDownloadWorker` define `finished = Signal(str)` (workers.py:54, 105, 150, 244), shadowing `QThread.finished`; thread-end cleanup cannot use it and callers cannot tell thread end from "result ready".
2. **No cancel for single video swaps** (no stop; `swap_video` has no cancellation hook), and `closeEvent` waits only for batch/webcam/models (main_window.py:909-919); closing mid-swap risks destroying a running QThread.
3. **Webcam.** `cap.release()` is not in a `finally` (workers.py:232); no `CAP_DSHOW` or buffer-size setting; unbounded queued frame signals; model load happens inside `run()` with only an indeterminate bar.
4. **State handling.** Output path derived next to the target and overwritten silently, may be read-only (main_window.py `_start_swap`); Score unsupported for video; every swap reloads all models (B1.1).
5. **Per-frame status signals** from the video callback (workers.py:120-122) flood the event loop.

### B4. Testing / packaging
1. **No model-backed tests.** Tests cover config, math helpers and mocks (e.g. `test_seamless.py` asserts preset fields and OpenCV enhancer only). Nothing runs HyperSwap/GFPGAN/XSeg on a golden image; no ID/sharpness regression gate in CI. `scripts/live_regression.py` is manual.
2. **CI.** Matrix is py3.9-3.12 while README/CHANGELOG advertise 3.13; `-x` aborts at first failure; integration job duplicates unit job; no Windows/DML/offscreen-Qt job; black 88 vs flake8 120; mypy, native build and docker are `continue-on-error`.
3. **Dependency conflicts.** Core deps include `torch`, `torchvision`, `mediapipe`, `onnxruntime` (CPU) while extras add `onnxruntime-gpu` / `-directml` (pyproject.toml:44-77): two ORT distributions side by side is a known break. `torch` is a multi-GB core dep for what is an ORT pipeline (only optional RAFT/training need it). `insightface` lacks wheels on newer Pythons and needs a compiler on Windows.
4. **Licence mismatch.** Code is MIT, but default weights are non-commercial: InsightFace `buffalo_l` and `inswapper_128` ([InsightFace](https://www.insightface.ai/en/solutions/face-swapping)), HyperSwap (ResearchRAIL per repo comment), CodeFormer (NTU S-Lab). README says only "review before commercial use"; the MIT badge and classifier imply a freely commercial SDK.
5. **Frozen build.** `./models` relative paths (pipeline.py:174) vs GUI `paths.py`; PyInstaller bundle includes torch; no signed installer.
6. **Docs drift.** README claims audio remux (B2.10); CHANGELOG 0.3.1 claims "restore ORT models forced to CPU when swapper is on DirectML" but no such code exists (the only CPU-force hit is the `--cpu` flag in webcam_demo.py:285).

---

## C. Algorithm landscape (late 2026) and recommendations

Sourcing note: web search results were partly secondary or vendor blogs. Items marked (unverified) need confirmation against the upstream repo before a decision.

### C.1 Detection
| Model | Quality | Speed | Licence | Take |
|---|---|---|---|---|
| SCRFD (`det_10g` in buffalo_l) | strong on WIDER, 5 kps | fast, ORT-friendly | weights non-commercial ([InsightFace](https://insightface.ai)) | keep default for research behind a `Detector` protocol |
| RetinaFace | similar, heavier | slower | non-commercial weights | drop; wrapper is buggy |
| YOLOv8/11-face | very good on small/occluded | fast with TRT | AGPL-3.0 via Ultralytics, enterprise licence for commercial ([source](https://dr.molodetz.nl/_proxy/https/github.com/yakhyo/uniface)) | optional plugin only |
| YuNet (OpenCV Zoo) | good frontal, 5 kps | very fast on CPU | Apache-2.0 | add as commercial-safe CPU detector |
| MediaPipe BlazeFace | good close-up webcam | very fast | Apache-2.0 ([ONNX bundle](https://huggingface.co/Heliosoph/mediapipe-face-onnx)) | add for realtime tracker |

### C.2 Landmarks
2d106: good but bundled and costly; 1k3d68: only for pose; MediaPipe FaceMesh 468: best jaw/contour for mask hulls, Apache-2.0, fast; 5-pt kps suffice for alignment. **Recommend:** compute dense landmarks only when a mask feature needs them; build paste masks from a mesh hull instead of YCrCb/luminance heuristics; smooth kps with One-Euro ([Casiez et al.](https://gery.casiez.net/1euro/), cited from memory, not re-fetched).

### C.3 Swappers
| Model | Res | Notes | Licence |
|---|---|---|---|
| InSwapper 128 | 128 | decent, soft, fast | non-commercial ([InsightFace](https://www.insightface.ai/en/solutions/face-swapping)) |
| HyperSwap 1a/1b/1c | 256 | FaceFusion default in 3.9 per a [setup guide](https://magichour.ai/blog/how-to-use-facefusion); better detail than inswapper in community use | ResearchRAIL per repo comment, not independently verified |
| SimSwap(-256) | 256 | older, ID leakage | non-commercial per [vendor table](https://www.wavespeed.ai/blog/posts/open-source-face-swap-software/) |
| GHOST 1/2/3 | 256 | good attribute preservation | Apache-2.0 claimed by a third-party wrapper (unverified) |
| BlendSwap, UniFace, ReSwapper | 128-256 | FaceFusion options / open inswapper re-impl | unverified |
| Diffusion: [Face-Adapter](https://arxiv.org/html/2405.12970v1), DreamID (SD-Turbo based, [review](https://themoonlight.io/en/review/dreamid-high-fidelity-and-fast-diffusion-based-face-swapping-via-triplet-id-group-learning)), IP-Adapter + ControlNet inpaint ([report](https://www.themoonlight.io/de/review/face-swap-via-diffusion-model)) | 512+ | best realism on stills, handles hair and glasses; seconds per image | mixed, mostly research. REFace and DiffSwap were not confirmed by search |

**Recommend:** HyperSwap-256 as quality default; add GHOST-256 (verify licence) as the commercial-safe pluggable swapper; keep InSwapper for speed; diffusion "studio mode" for stills only, post-v1. Move per-model facts (input size, normalisation, latent type, licence) into a `SwapperSpec` in the registry.

### C.4 Occlusion / parsing
XSeg (in use; fast; 256), BiSeNet face parsing (19 classes incl. hair, glasses, neck; fast; check weight licence), SAM 2 (Apache-2.0, [repo](https://github.com/facebookresearch/sam2); README lists tens of FPS for the tiny model on server GPUs, figures vary by version). **Recommend:** run XSeg on the **aligned 256 crop** and warp the mask back with the swap affine (fold into the paste mask); BiSeNet optional for hair/neck; SAM 2 only for offline rotoscope refinement.

### C.5 Restorers
GFPGAN 1.4: fast, can drift identity/hue, Apache-2.0 ([source](https://aireiter.com/blog/gfpgan-codeformer-face-restoration)). CodeFormer: best fidelity knob, NTU S-Lab non-commercial ([source](https://gitee.com/sieding/CodeFormer)). GPEN-BFR 512, RestoreFormer++: licences unverified. DiffBIR/PMRF: highest quality, seconds per face, research. **Recommend:** keep GFPGAN/GPEN, default blend 0.3-0.5, apply to the **aligned** crop (reuse the swapper's norm_crop at the 512 template) rather than tiled bbox crops, skip in realtime, size-gate; diffusion restorers offline only.

### C.6 Temporal
RAFT (accurate, heavy; use `raft_small`, cached, ROI only), Farneback (cheap, noisy), EMA on kps (laggy), Kalman / One-Euro on landmarks (adaptive lag vs jitter). **Recommend:** One-Euro on kps/bbox, flow only inside the face ROI (or none), per-track latent EMA for identity mix, mask-edge hysteresis instead of a global frame blend.

### C.7 ID and quality metrics
Keep ArcFace cosine but note it is optimistic (same family as the generating embedding); add an independent embedder (e.g. AdaFace, unverified here), LPIPS on non-face regions to detect background damage, landmark/pose RMSE for expression preservation, a warp-error flicker metric, and sharpness gain. Gate CI on a golden-pair regression with thresholds.

---

## D. Speed plan (all gains are estimates, not measurements)

Working assumption: at 720p-1080p with `seamless`, time is dominated by (a) restorer tiles (9 GFPGAN-512 runs per face), (b) full-frame CPU passes (paste, copies, flow), (c) per-call H2D/D2H and CPU pre/post. Detection plus swap on a mid-range GPU is likely a minority. Measure first (item D15).

| Step | Change | Rationale | Estimated gain |
|---|---|---|---|
| D0 | Session/pipeline cache (LRU), model-hash stamp | removes multi-second to tens-of-seconds model load per call | large for short jobs and GUI re-runs, none for long video |
| D1 | Fix tile schedule (9 -> 4), then single aligned-512 restore pass | 9 -> 1 restorer run per face | restore stage roughly 5-9x faster |
| D2 | ROI paste-back, in-place ROI ops, delete dead `fake_diff` | O(face area) instead of O(frame area) | CPU compositing roughly 3-10x at 1080p |
| D3 | Always prune buffalo_l to detection (+ recognition only if needed) | drops 3 of 5 models per detect | detect stage roughly 1.5-2.5x |
| D4 | ORT provider options: CUDA `cudnn_conv_algo_search`, `arena_extend_strategy`, shared allocator, `device_id`; DML mem-pattern off and sequential | stability, VRAM | modest, mainly reliability |
| D5 | **IOBinding** with preallocated device buffers and reused `OrtValue`s; keep crop tensors on device ([ORT API](https://onnxruntime.ai/docs/api/python/api_summary.html)) | removes per-call host copies and allocs | roughly 1.2-1.5x on small-model-bound stages |
| D6 | **CUDA graphs** (`enable_cuda_graph`) for fixed-shape models (swapper 256, restorer 512, XSeg 256); needs static IOBinding ([NVIDIA blog](https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-cuda-and-tensorrt-execution-providers-in-onnx-runtime/)) | cuts launch overhead of small kernels | roughly 1.2-2x on those models |
| D7 | **TensorRT EP**: `trt_fp16_enable`, `trt_engine_cache_enable`, cache per GPU compute capability + ORT/TRT version ([ORT TRT docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)); INT8 only after calibration and a per-model quality check (static quantisation required for TRT per [Optimum docs](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu)) | FP16 tensor cores | roughly 1.5-3x per model vs CUDA FP32; first-run engine build takes minutes, warm in background. Delete or rewrite the unused pycuda runtime on the TRT 10 API |
| D8 | DirectML (AMD/Intel): tuned options, FP16 models; CoreML EP on Apple with compiled-model cache | portability | no claim without measurement |
| D9 | **Frames on GPU**: NVDEC decode (PyAV / ffmpeg `-hwaccel cuda`), torch or CuPy tensors for affine crop (`grid_sample`), colour match, mask blend, paste; NVENC encode (`h264_nvenc`). Detection post-processing stays on CPU | removes 1080p BGR CPU round trips | the largest single win for 1080p+ video, roughly 2-4x end to end; needs LAB, Gaussian blur and Reinhard reimplemented in torch |
| D10 | **Pipelined stages** with bounded ring buffers: decode -> detect/track -> swap+restore (GPU) -> composite -> encode threads (ORT and cv2 release the GIL) | overlap CPU and GPU | roughly 1.5-2.5x offline throughput; bounded latency live by dropping stale frames |
| D11 | **Batching**: offline, batch N frames/faces per swapper/restorer call | amortise launch | roughly 1.3-2x on those stages |
| D12 | Multi-stream: separate CUDA streams for detect(t+1) overlapping swap(t) | latency hiding | roughly 1.2-1.4x |
| D13 | Warp/paste on GPU (`grid_sample`, separable Gaussian) | removes CPU compositing | folded into D9 |
| D14 | Realtime tracker: detect every N + One-Euro/Kalman + ROI landmarker; SCRFD only on loss | detection is the known live bottleneck (PRD ref [Deep-Live-Cam #1664](https://github.com/hacksider/Deep-Live-Cam/issues/1664)) | detection amortised roughly 3-5x |
| D15 | Per-stage profiler with CUDA events; publish benchmark suite | replaces these estimates with data | n/a |

**Headline:** Phase 1 (D0-D3) are software fixes with the best gain per effort. Phase 2 (D4-D7, D10) adds GPU-native inference and pipelining. Phase 3 (D9, D11-D13) moves the whole frame path to GPU. The PRD target (<= 40 ms at 720p on RTX 30/40) looks plausible only after D3 + D5-D7 + D9/D10 with the restorer off in realtime.

---

## E. DSA mapping

| Structure | Where | Stage | Why |
|---|---|---|---|
| Bounded **ring buffer** (preallocated slots; drop-oldest live, block offline) | new `face_swap/runtime/ring.py`; used by `api.swap_video`, `start_realtime_swap`, `WebcamWorker` | decode -> detect -> swap -> encode | no per-frame alloc, backpressure; replaces unbounded Qt signal queue (workers.py:231) |
| **Priority queue** (heap by priority/deadline) | new `face_swap/runtime/jobs.py`; GUI `BatchWorker` | batch jobs vs interactive preview | preview preempts batch; fair server use |
| **IoU tracker + Hungarian assignment** (`scipy.optimize.linear_sum_assignment`, already a dependency) with per-track One-Euro/Kalman and TTL | new `face_swap/tracking/tracker.py`; replaces EMA matching (smoother.py:342-351) and index select (adaptive.py:348-355) | between detect and select | fixes B2.3-B2.4; stable identity; per-track source mapping |
| **One-Euro filter** (x_prev, dx_prev, t_prev per coordinate) | `temporal/smoother.py` | kps/bbox, mask corners | adaptive lag vs jitter |
| **LRU cache** (OrderedDict, VRAM-aware) | `core/session_cache.py`; keys (model sha, providers, options); source-embedding cache keyed by image hash | model load, source embed | fixes B1.1 |
| **KD-tree / FAISS** (flat or HNSW) over normalised 512-d embeddings | `embedding/gallery.py` | identity-based routing: map target people to sources, cluster faces in a video | cosine = inner product; FAISS only above ~1e4 vectors, else a numpy matmul |
| **Interval tree** (or sorted list + bisect) | `core/timeline.py` | per-segment config (faces/sources/quality between t0..t1), GUI timeline | O(log n + k) lookup per frame |
| **Content-hash dedupe + stamp file** (sha256 -> path; stamp with size+mtime; HTTP Range resume) | `core/model_manager.py` | model download/verify | avoids re-hashing 400 MB per load, dedupes mirrors, resumable |
| **Sliding-window deque** | profiler, flicker metric, fps | profiling, temporal metrics | O(1) updates |
| **Spatial hash grid** | crowd videos | track candidate gating | avoids O(n*m) IoU at high face counts |

---

## F. Prioritised implementation plan

Effort: S <= 1 day, M = 2-4 days, L = 1-2 weeks, XL > 2 weeks.

### P0 (correctness and cheap big wins)
1. Session/pipeline cache; build the pipeline once per GUI session or API handle (B1.1; E LRU). **M**
2. Fix `_tile_starts`; stop tiling unaligned bbox crops; restore on the aligned 512 crop and warp back; rename "pixel boost" semantics (B1.4, B2.1). **M**
3. ROI paste-back, remove dead `fake_diff`, in-place ROI compositing, ROI-sized hull masks (B1.2-B1.6). **M**
4. Wire audio mux (ffmpeg/PyAV) and replace `mp4v` with H.264 at sane CRF; correct the README claim until done (B2.10). **M**
5. Replace the global flow blend with ROI-only or remove it; cache RAFT; add One-Euro on kps (B2.2-B2.3, B1.12). **M**
6. Fix or drop `RetinaFaceDetector` BGR bug and stop instantiating modular parts in the native path (B2.8). **S**
7. GUI: rename `finished` signals, add video cancel, release camera in `finally`, `closeEvent` waits for swap (B3). **S**
8. Unicode-safe imread/imwrite, correct error messages, guard fps/writer (B2.13). **S**
9. Licence honesty: per-model licence table, a commercial-safe preset, README/classifier update (B4.4). **S**
10. Golden-image regression test (ID, sharpness, LPIPS thresholds); drop `-x`; add Windows and offscreen-Qt jobs (B4.1-B4.2). **L**

### P1 (GPU performance and structure)
1. ORT provider-options helper; DML mem-pattern/sequential fix; shared arena (D4). **M**
2. IOBinding + CUDA graphs for swapper/restorer/XSeg (D5, D6). **L**
3. TensorRT EP with FP16 and engine cache; delete or rewrite `optimization/runtime.py` (D7). **L**
4. Threaded pipeline with ring buffers; bounded-latency webcam mode (D10, E). **L**
5. IoU + Hungarian tracker; stable multi-face identity; per-track source mapping (E). **L**
6. Prune buffalo_l modules, lazy embedding, cap the detect ladder (B1.9-B1.10). **S**
7. XSeg on the aligned crop merged into the paste mask (B2.6). **M**
8. Dependency split: ORT only via extras (`cpu|gpu|dml`), torch optional, conflict check in `first_run_check`, Python 3.13 insightface path (B4.3). **M**
9. Add GHOST-256 or another permissive swapper via a `SwapperSpec` registry (C.3). **L**

### P2 (larger bets)
1. Full GPU frame path: NVDEC/NVENC, torch/CuPy compositing (D9, D13). **XL**
2. Batched offline inference and multi-stream (D11, D12). **L**
3. MediaPipe-mesh-based masks replacing skin/beard heuristics, with a fairness test set across skin tones (B2.7). **L**
4. FAISS/KD gallery and identity clustering for multi-person videos (E). **M**
5. Robust watermark (QIM or learned, or C2PA metadata), default-on for non-research profiles (B2.9). **L**
6. Diffusion "studio" mode for stills (Face-Adapter/DreamID class) plus a diffusion restorer. **XL**
7. CoreML/Apple and mobile paths: wire up with tests or remove (B2.15). **L**
8. Per-stage profiler with GPU timings and a published benchmark suite to replace section D estimates (D15). **M**
