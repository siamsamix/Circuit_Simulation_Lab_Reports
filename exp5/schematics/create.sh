#!/bin/bash
netlist_dir="/mnt/G/Circuit Simulation Laboratory/exp5/netlist"
schem_dir="/mnt/G/Circuit Simulation Laboratory/exp5/schematics"
py="/mnt/G/Circuit Simulation Laboratory/spice/bin/python"
script="/mnt/G/Circuit Simulation Laboratory/netlist2schematic.py"

shopt -s nullglob

for f in "$netlist_dir"/*; do
    [[ -f "$f" ]] || continue

    # snapshot of netlist dir before running
    before=$(ls -A "$netlist_dir")

    "$py" "$script" "$f" --theme capture

    # move any new files (except the source netlist) into schematics/
    for out in "$netlist_dir"/*; do
        [[ -f "$out" ]] || continue
        [[ "$out" == "$f" ]] && continue
        grep -qxF "$(basename "$out")" <<<"$before" && continue
        mv -v "$out" "$schem_dir/"
    done
done
