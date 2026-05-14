#!/usr/bin/env bash
set -euo pipefail

echo "=== Ruff: lint ==="
ruff check src scripts

echo
echo "=== Ruff: format check ==="
ruff format --check src scripts

echo
echo "=== Mypy ==="
mypy src

echo
echo "All checks passed."
