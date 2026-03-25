cmake_minimum_required(VERSION 3.18)
project(VolCore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Release flags: maximum performance
add_compile_options(
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:Release>:-ffast-math>
    $<$<CONFIG:Release>:-DNDEBUG>
)

# ── pybind11 (download if not installed) ─────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

# ── Python extension module: vol_core ────────────────────────────────────────
pybind11_add_module(vol_core src/bindings.cpp)
target_include_directories(vol_core PRIVATE src)

# Output the .so next to the notebooks so `import vol_core` works without install
set_target_properties(vol_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ── C++ unit tests (requires Catch2 v3) ──────────────────────────────────────
find_package(Catch2 3 QUIET)
if(Catch2_FOUND)
    add_executable(test_greeks tests/test_greeks.cpp)
    target_include_directories(test_greeks PRIVATE src)
    target_link_libraries(test_greeks PRIVATE Catch2::Catch2WithMain)
    include(CTest)
    include(Catch)
    catch_discover_tests(test_greeks)
    message(STATUS "Catch2 found — unit tests enabled")
else()
    message(STATUS "Catch2 not found — skipping unit tests (install with: brew install catch2)")
endif()

# ── Microbenchmarks (requires Google Benchmark) ───────────────────────────────
find_package(benchmark QUIET)
if(benchmark_FOUND)
    add_executable(bench_surface benchmarks/bench_surface.cpp)
    target_include_directories(bench_surface PRIVATE src)
    target_link_libraries(bench_surface PRIVATE benchmark::benchmark benchmark::benchmark_main)
    message(STATUS "Google Benchmark found — C++ benchmarks enabled")
else()
    message(STATUS "Google Benchmark not found — skipping C++ benchmarks")
    message(STATUS "Install with: brew install google-benchmark  or  vcpkg install benchmark")
endif()
