#!/usr/bin/env bash
# Select built targets from the install receipt's "  component:target" rows.
set -euo pipefail
case "$2" in
    host) components='core|tools' ;;
    hip) components='hip' ;;
    cuda-qwen) components='cuda' ;;
    *) exit 2 ;;
esac
sed -En "s/^  ($components):([A-Za-z0-9_.-]+)\$/\\2/p" "$1"
