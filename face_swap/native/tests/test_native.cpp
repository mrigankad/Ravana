/**
 * @file test_native.cpp
 * @brief Minimal smoke test for the Face Swap C API.
 *
 * Build with:
 *   cmake -B build -S . -DBUILD_TESTS=ON && cmake --build build
 *
 * Run:
 *   ./build/fs_test_native
 */

#include "face_swap.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#define CHECK(expr)                                              \
    do {                                                         \
        FsError _err = (expr);                                   \
        if (_err != FS_OK) {                                     \
            fprintf(stderr, "FAIL: %s → %s\n", #expr,           \
                    fs_error_string(_err));                       \
            return 1;                                             \
        }                                                        \
    } while (0)

static int test_init_shutdown() {
    printf("  test_init_shutdown ... ");
    CHECK(fs_init());
    CHECK(fs_shutdown());
    printf("OK\n");
    return 0;
}

static int test_version() {
    printf("  test_version ... ");
    const char* v = fs_version();
    assert(v != nullptr);
    assert(strlen(v) > 0);
    printf("OK (%s)\n", v);
    return 0;
}

static int test_error_string() {
    printf("  test_error_string ... ");
    const char* s = fs_error_string(FS_OK);
    assert(s != nullptr);
    assert(strcmp(s, "Success") == 0);

    s = fs_error_string(FS_ERROR_NO_FACE);
    assert(s != nullptr);
    printf("OK\n");
    return 0;
}

static int test_config_default() {
    printf("  test_config_default ... ");
    FsConfig cfg;
    CHECK(fs_config_default(&cfg));
    assert(cfg.device == FS_DEVICE_CUDA);
    assert(cfg.quality == FS_QUALITY_HIGH);
    assert(cfg.crop_size == 256);
    assert(cfg.color_correction == 1);
    printf("OK\n");
    return 0;
}

static int test_session_lifecycle() {
    printf("  test_session_lifecycle ... ");
    CHECK(fs_init());

    FsConfig cfg;
    CHECK(fs_config_default(&cfg));
    cfg.device = FS_DEVICE_CPU;  // Use CPU for test

    FsSession session = nullptr;
    CHECK(fs_session_create(&session, &cfg));
    assert(session != nullptr);

    CHECK(fs_session_destroy(session));
    CHECK(fs_shutdown());
    printf("OK\n");
    return 0;
}

static int test_null_guards() {
    printf("  test_null_guards ... ");
    assert(fs_config_default(nullptr) == FS_ERROR_INVALID_ARG);
    assert(fs_session_create(nullptr, nullptr) == FS_ERROR_INVALID_ARG);
    assert(fs_session_destroy(nullptr) == FS_ERROR_INVALID_ARG);
    assert(fs_session_set_source_file(nullptr, nullptr) == FS_ERROR_INVALID_ARG);
    printf("OK\n");
    return 0;
}

int main() {
    int failures = 0;

    printf("Running native C API tests:\n");

    failures += test_init_shutdown();
    failures += test_version();
    failures += test_error_string();
    failures += test_config_default();
    failures += test_session_lifecycle();
    failures += test_null_guards();

    printf("\n");
    if (failures == 0) {
        printf("All tests passed!\n");
    } else {
        printf("%d test(s) FAILED!\n", failures);
    }

    return failures;
}
