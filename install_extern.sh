#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TQO_DIR="$SCRIPT_DIR/extern/TensorQuadOperation"


echo "==> Cleaning extern/ ..."
rm -rf "$SCRIPT_DIR"/extern/*

echo "==> Initializing submodules..."
git -C "$SCRIPT_DIR" submodule update --init --recursive

echo "==> install TensorQuadOperation:"
cd "$TQO_DIR"
bash install_extern.sh

echo "==> Done!"