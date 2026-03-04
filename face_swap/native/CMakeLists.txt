cmake_minimum_required(VERSION 3.18)
project(face_swap_native VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Options ─────────────────────────────────────────────────────────────
option(WITH_ONNXRUNTIME   "Build with ONNX Runtime support"  ON)
option(WITH_OPENCV        "Build with OpenCV support"        ON)
option(BUILD_SHARED_LIBS  "Build as shared library (.dll/.so)" ON)

# ── Sources ─────────────────────────────────────────────────────────────
set(SOURCES
    src/face_swap_c.cpp
)

set(HEADERS
    include/face_swap.h
)

# ── Library target ──────────────────────────────────────────────────────
add_library(face_swap ${SOURCES} ${HEADERS})

target_include_directories(face_swap
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(BUILD_SHARED_LIBS)
    target_compile_definitions(face_swap PRIVATE FACE_SWAP_BUILD_DLL)
endif()

# ── ONNX Runtime ────────────────────────────────────────────────────────
if(WITH_ONNXRUNTIME)
    find_package(onnxruntime QUIET)
    if(onnxruntime_FOUND)
        target_link_libraries(face_swap PRIVATE onnxruntime::onnxruntime)
        target_compile_definitions(face_swap PRIVATE HAS_ONNXRUNTIME)
    else()
        # Fallback: look for manual ORT path
        if(DEFINED ORT_ROOT)
            target_include_directories(face_swap PRIVATE ${ORT_ROOT}/include)
            target_link_directories(face_swap PRIVATE ${ORT_ROOT}/lib)
            target_link_libraries(face_swap PRIVATE onnxruntime)
            target_compile_definitions(face_swap PRIVATE HAS_ONNXRUNTIME)
        else()
            message(WARNING "ONNX Runtime not found. Set -DORT_ROOT=<path> or -DWITH_ONNXRUNTIME=OFF")
        endif()
    endif()
endif()

# ── OpenCV ──────────────────────────────────────────────────────────────
if(WITH_OPENCV)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
    target_link_libraries(face_swap PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(face_swap PRIVATE HAS_OPENCV)
    target_include_directories(face_swap PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

# ── Install ─────────────────────────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS face_swap
    EXPORT face_swap-targets
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/face_swap)

install(EXPORT face_swap-targets
    FILE face_swap-targets.cmake
    NAMESPACE face_swap::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/face_swap
)

# ── Simple test executable ──────────────────────────────────────────────
option(BUILD_TESTS "Build test executable" OFF)
if(BUILD_TESTS)
    add_executable(fs_test_native tests/test_native.cpp)
    target_link_libraries(fs_test_native PRIVATE face_swap)
endif()
