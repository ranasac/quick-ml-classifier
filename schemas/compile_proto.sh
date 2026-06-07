#!/usr/bin/env bash
# Compile the transaction.proto file into Python bindings.
# Run from the repository root:  bash schemas/compile_proto.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$SCRIPT_DIR/generated"

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/__init__.py"

echo "Compiling transaction.proto → $OUT_DIR"
protoc \
  --proto_path="$SCRIPT_DIR" \
  --python_out="$OUT_DIR" \
  "$SCRIPT_DIR/transaction.proto"

echo "Done. Generated files:"
ls "$OUT_DIR"
