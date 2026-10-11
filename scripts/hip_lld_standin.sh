#!/bin/sh
# compile-only stand-in for ROCm's ld.lld: copies the assembled object to -o. The result is NOT a loadable code object;
# it lets triton finish compiling for an AMD target on a machine without ROCm, so the assembler's diagnostics are seen.
[ "$1" = "--version" ] && { echo "LLD 0.0 (compile-only stand-in: copies its input; not a linker)"; exit 0; }
in=""; out=""
while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift 2;; -flavor) shift 2;; -*) shift;; *) in="$1"; shift;; esac; done
cp "$in" "$out"
