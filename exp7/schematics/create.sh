#!/bin/bash
# Draw every netlist in ../netlist as a capture-style schematic (png) in this folder.
# Usage: ./create.sh            (needs: pip install schemdraw matplotlib)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
netlist_dir="$here/../netlist"
script="$here/../../netlist2schematic.py"
py="../../spice/bin/python3"

shopt -s nullglob
for f in "$netlist_dir"/*.cir; do
    b="$(basename "$f" .cir)"
    "$py" "$script" "$f" --theme capture -o "$here/$b.png"
done
