#!/usr/bin/env bash
# Fetch the upstream LIBERO benchmark into ./LIBERO.
# The Modal images mount ./LIBERO into the container, so run this once from the
# repo root before any `modal run`. LIBERO is used unmodified.
set -euo pipefail
if [ -d LIBERO ]; then
  echo "LIBERO/ already exists; skipping."
  exit 0
fi
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
echo "LIBERO cloned into ./LIBERO"
