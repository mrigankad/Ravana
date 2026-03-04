/**
 * @file face_swap.h
 * @brief C API for Face Swap SDK - Real-Time Face Swapping System
 *
 * As per PRD Section 4.1, this provides a C API suitable for integration
 * into native apps (C, C++, and any language with C FFI support).
 *
 * Usage:
 *   1. Call fs_init() once at startup.
 *   2. Create a session with fs_session_create().
 *   3. Load a source identity with fs_session_set_source().
 *   4. Process frames with fs_session_swap_image() or fs_session_swap_frame().
 *   5. Destroy the session with fs_session_destroy().
 *   6. Call fs_shutdown() at exit.
 *
 * Thread Safety:
 *   - fs_init/fs_shutdown are NOT thread-safe; call from main thread only.
 *   - Each FsSession is independent and NOT thread-safe internally.
 *   - Multiple sessions CAN be used from different threads simultaneously.
 *
 * @copyright MIT License
 */

#ifndef FACE_SWAP_H
#define FACE_SWAP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Build configuration
 * ----------------------------------------------------------------------- */

#ifdef _WIN32
#   ifdef FACE_SWAP_BUILD_DLL
#       define FS_API __declspec(dllexport)
#   else
#       define FS_API __declspec(dllimport)
#   endif
#else
#   define FS_API __attribute__((visibility("default")))
#endif

/* --------------------------------------------------------------------------
 * Error codes
 * ----------------------------------------------------------------------- */

typedef enum FsError {
    FS_OK                   = 0,
    FS_ERROR_INVALID_ARG    = -1,
    FS_ERROR_NOT_INIT       = -2,
    FS_ERROR_MODEL_LOAD     = -3,
    FS_ERROR_NO_FACE        = -4,
    FS_ERROR_SWAP_FAILED    = -5,
    FS_ERROR_OUT_OF_MEMORY  = -6,
    FS_ERROR_IO             = -7,
    FS_ERROR_UNSUPPORTED    = -8,
    FS_ERROR_INTERNAL       = -99,
} FsError;

/* --------------------------------------------------------------------------
 * Opaque handles
 * ----------------------------------------------------------------------- */

/** Opaque session handle. */
typedef struct FsSession_T* FsSession;

/* --------------------------------------------------------------------------
 * Configuration structures
 * ----------------------------------------------------------------------- */

/** Device type. */
typedef enum FsDevice {
    FS_DEVICE_CPU  = 0,
    FS_DEVICE_CUDA = 1,
} FsDevice;

/** Quality preset. */
typedef enum FsQuality {
    FS_QUALITY_LOW    = 0,
    FS_QUALITY_MEDIUM = 1,
    FS_QUALITY_HIGH   = 2,
} FsQuality;

/** Blend mode. */
typedef enum FsBlendMode {
    FS_BLEND_ALPHA   = 0,
    FS_BLEND_POISSON = 1,
    FS_BLEND_FEATHER = 2,
} FsBlendMode;

/** Pixel format of image buffers. */
typedef enum FsPixelFormat {
    FS_PIXEL_BGR  = 0,   /**< OpenCV default (Blue-Green-Red) */
    FS_PIXEL_RGB  = 1,
    FS_PIXEL_BGRA = 2,
    FS_PIXEL_RGBA = 3,
} FsPixelFormat;

/** Session configuration (pass to fs_session_create). */
typedef struct FsConfig {
    FsDevice      device;
    FsQuality     quality;
    FsBlendMode   blend_mode;
    int           color_correction;     /**< 0 = off, 1 = on */
    int           enable_temporal;      /**< 0 = off, 1 = on */
    int           async_detection;      /**< 0 = off, 1 = on */
    int           enable_watermark;     /**< 0 = off, 1 = on */
    int           crop_size;            /**< 128, 256, or 512 */
    const char*   swap_model_path;      /**< NULL for default */
    const char*   detection_model_path; /**< NULL for default */
} FsConfig;

/** Bounding box returned by detection. */
typedef struct FsBBox {
    float x1, y1, x2, y2;
    float confidence;
    int   track_id;
} FsBBox;

/** Image buffer descriptor (non-owning view). */
typedef struct FsImage {
    const uint8_t* data;
    int            width;
    int            height;
    int            stride;        /**< Bytes per row (0 = width * channels) */
    FsPixelFormat  pixel_format;
} FsImage;

/** Mutable image buffer (caller-owned). */
typedef struct FsImageMut {
    uint8_t*       data;
    int            width;
    int            height;
    int            stride;
    FsPixelFormat  pixel_format;
} FsImageMut;

/** Per-frame timing report. */
typedef struct FsTimings {
    float detection_ms;
    float landmarks_ms;
    float alignment_ms;
    float swap_ms;
    float blend_ms;
    float total_ms;
    int   num_faces;
} FsTimings;

/* --------------------------------------------------------------------------
 * Library lifecycle
 * ----------------------------------------------------------------------- */

/**
 * Initialise the Face Swap library.
 * Must be called once before any other fs_* function.
 */
FS_API FsError fs_init(void);

/**
 * Shut down the library and release all global resources.
 * After this call no other fs_* function may be used.
 */
FS_API FsError fs_shutdown(void);

/**
 * Get a human-readable string for an error code.
 */
FS_API const char* fs_error_string(FsError err);

/**
 * Get the library version string (e.g. "0.1.0").
 */
FS_API const char* fs_version(void);

/* --------------------------------------------------------------------------
 * Default configuration
 * ----------------------------------------------------------------------- */

/**
 * Fill *cfg with sensible defaults.
 *
 * Defaults: CUDA, high quality, alpha blend, colour correction on,
 * temporal on, no watermark, crop 256.
 */
FS_API FsError fs_config_default(FsConfig* cfg);

/* --------------------------------------------------------------------------
 * Session management
 * ----------------------------------------------------------------------- */

/**
 * Create a swap session.
 *
 * @param[out] session  Receives the new session handle.
 * @param[in]  config   Configuration (may be NULL for defaults).
 * @return FS_OK on success.
 */
FS_API FsError fs_session_create(FsSession* session, const FsConfig* config);

/**
 * Destroy a session and release all resources.
 */
FS_API FsError fs_session_destroy(FsSession session);

/* --------------------------------------------------------------------------
 * Source identity
 * ----------------------------------------------------------------------- */

/**
 * Set the source identity from an image file (PNG/JPEG).
 *
 * @param session   Active session.
 * @param path      Path to source image file.
 */
FS_API FsError fs_session_set_source_file(FsSession session, const char* path);

/**
 * Set the source identity from an in-memory image buffer.
 *
 * @param session   Active session.
 * @param image     Source image buffer.
 */
FS_API FsError fs_session_set_source(FsSession session, const FsImage* image);

/**
 * Set the source identity by averaging multiple images.
 *
 * @param session   Active session.
 * @param images    Array of image buffers.
 * @param count     Number of images.
 */
FS_API FsError fs_session_set_source_multi(
    FsSession session,
    const FsImage* images,
    int count
);

/* --------------------------------------------------------------------------
 * Face detection (low-level)
 * ----------------------------------------------------------------------- */

/**
 * Detect faces in an image.
 *
 * @param session      Active session.
 * @param image        Input image.
 * @param[out] bboxes  Caller-allocated array to receive bounding boxes.
 * @param[in,out] count  In: capacity of bboxes array. Out: actual count.
 */
FS_API FsError fs_detect_faces(
    FsSession session,
    const FsImage* image,
    FsBBox* bboxes,
    int* count
);

/* --------------------------------------------------------------------------
 * Face swapping
 * ----------------------------------------------------------------------- */

/**
 * Swap face on a single image (high-level).
 *
 * Source identity must have been set via fs_session_set_source*().
 *
 * @param session   Active session.
 * @param target    Target image to swap face onto.
 * @param[out] out  Output image buffer (caller allocates data,
 *                  must be at least target.width * target.height * channels bytes).
 */
FS_API FsError fs_swap_image(
    FsSession session,
    const FsImage* target,
    FsImageMut* out
);

/**
 * Swap face on a single image read from / written to file.
 */
FS_API FsError fs_swap_image_file(
    FsSession session,
    const char* target_path,
    const char* output_path
);

/**
 * Process a single video frame (includes temporal smoothing).
 *
 * @param session       Active session.
 * @param frame         Input frame.
 * @param frame_number  Sequential frame index (for temporal state).
 * @param[out] out      Output frame buffer.
 * @param[out] timings  Optional timing report (may be NULL).
 */
FS_API FsError fs_swap_frame(
    FsSession session,
    const FsImage* frame,
    int frame_number,
    FsImageMut* out,
    FsTimings* timings
);

/**
 * Process an entire video file.
 *
 * @param session          Active session.
 * @param input_path       Path to input video (MP4/MOV).
 * @param output_path      Path to write output video.
 * @param progress_cb      Optional callback(frame_idx, total_frames, user_data).
 * @param user_data        Passed to progress_cb.
 */
FS_API FsError fs_swap_video(
    FsSession session,
    const char* input_path,
    const char* output_path,
    void (*progress_cb)(int frame_idx, int total_frames, void* user_data),
    void* user_data
);

/* --------------------------------------------------------------------------
 * Real-time webcam
 * ----------------------------------------------------------------------- */

/**
 * Start a real-time webcam swap loop.
 *
 * Blocks until the user presses 'q' in the preview window.
 *
 * @param session     Active session.
 * @param camera_id   Camera device index.
 * @param frame_cb    Optional per-frame callback (receives output frame).
 * @param user_data   Passed to frame_cb.
 */
FS_API FsError fs_start_realtime(
    FsSession session,
    int camera_id,
    void (*frame_cb)(const FsImage* frame, void* user_data),
    void* user_data
);

/* --------------------------------------------------------------------------
 * Benchmarking / profiling
 * ----------------------------------------------------------------------- */

/**
 * Get the latest per-frame timing breakdown.
 */
FS_API FsError fs_get_timings(FsSession session, FsTimings* out);

/**
 * Get average FPS over the last N frames.
 */
FS_API FsError fs_get_avg_fps(FsSession session, float* fps);

#ifdef __cplusplus
}
#endif

#endif /* FACE_SWAP_H */
