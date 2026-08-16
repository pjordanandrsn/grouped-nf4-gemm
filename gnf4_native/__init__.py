# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""gnf4_native — compile-at-first-use CPU kernels for the hybrid tier.

The C source ships as package data and is built on the target box with
``cc -O3 -march=native`` (ISA selection is a compile-time fact of the box
that runs it; a portable exact path is always present). See ``build.load``.
"""

from .build import available, features, load  # noqa: F401
