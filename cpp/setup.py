"""
pip-installable build for the vol_core C++ extension.

Usage
-----
    pip install ./cpp           # build + install (release mode)
    pip install -e ./cpp        # editable install (rebuilds on source change)

Requires pybind11:
    pip install pybind11

The extension is compiled with -O3 -march=native for maximum performance.
"""
from __future__ import annotations

import sys
from pathlib import Path

from pybind11.setup_helpers import build_ext, Pybind11Extension
from setuptools import setup

# Detect compiler flags
compile_flags = ["-O3", "-std=c++17"]
if sys.platform == "linux":
    compile_flags += ["-march=native", "-ffast-math"]
elif sys.platform == "darwin":
    compile_flags += ["-march=native"]

ext = Pybind11Extension(
    name="vol_core",
    sources=["src/bindings.cpp"],
    include_dirs=["src"],
    extra_compile_args=compile_flags,
    cxx_std=17,
)

setup(
    name="vol-core",
    version="0.1.0",
    description="Black-Scholes + IV solver + SVI surface C++ core with Python bindings",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=["pybind11>=2.11"],
)
