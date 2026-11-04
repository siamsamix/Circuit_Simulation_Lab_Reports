#!/bin/bash
#
# generate_graphs.sh
# Batch-run simulate_and_plot.py on every .cir in the netlist folder
# and write the resulting plots into this (graphs) folder.
#
# Usage:
#   ./create.sh                                   # default: x=time, y=v(1) v(2)
#   ./create.sh time v(1) v(2)                    # custom x + y vars
#   ./create.sh freq mag(v(out))                  # AC example
#   ./create.sh time 'v(1)-v(2)'                  # expression (quote it)
#
# Every run attaches framed notes to the traces: "Vin" on v(1), and on
# v(2) "Voltage across resistor" for the series clippers, "Voltage across
# diode" for the parallel (shunt) clippers, and "Output voltage" for the
# dual clippers.  Notes are only added for y vars that were requested.
#
# Env overrides (optional):
#   NETLIST_DIR, GRAPHS_DIR, PYTHON, SIM_SCRIPT

set -euo pipefail

NETLIST_DIR="${NETLIST_DIR:-/mnt/G/Circuit Simulation Laboratory/exp6/netlist}"
GRAPHS_DIR="${GRAPHS_DIR:-/mnt/G/Circuit Simulation Laboratory/exp6/graphs}"
PYTHON="${PYTHON:-/mnt/G/Circuit Simulation Laboratory/spice/bin/python}"
SIM_SCRIPT="${SIM_SCRIPT:-$GRAPHS_DIR/simulate_and_plot.py}"

# Non-interactive backend so plt.show() inside the Python script
# doesn't open/block on a window for every file in the loop.
export MPLBACKEND=Agg

# --- defaults, overridable via CLI ---
xvar="time"
yvars=("v(1)" "v(2)")

if [ $# -ge 1 ]; then
    xvar="$1"; shift
fi
if [ $# -ge 1 ]; then
    yvars=("$@")
fi

# --- sanity checks ---
[ -x "$PYTHON" ]     || { echo "Python not found: $PYTHON" >&2; exit 1; }
[ -f "$SIM_SCRIPT" ] || { echo "Sim script not found: $SIM_SCRIPT" >&2; exit 1; }
[ -d "$NETLIST_DIR" ]|| { echo "Netlist dir not found: $NETLIST_DIR" >&2; exit 1; }

shopt -s nullglob
cir_files=("$NETLIST_DIR"/*.cir)
if [ ${#cir_files[@]} -eq 0 ]; then
    echo "No .cir files found in $NETLIST_DIR" >&2
    exit 1
fi

echo "x-var : $xvar"
echo "y-vars: ${yvars[*]}"
echo

for cir in "${cir_files[@]}"; do
    base=$(basename "$cir" .cir)
    out="$GRAPHS_DIR/${base}.svg"

    # note for the v(2) trace, chosen by circuit: in the series clippers
    # the output node sits on the load resistor, in the shunt clippers it
    # sits on the diode, and in the dual clippers it is the clipped output
    note2="Output voltage"
    #case "$base" in
    #    shunt_*)      note2="Voltage across diode" ;;
    #    series_dual*) note2="Output voltage" ;;
    #esac

    # --note specs are keyed by trace name, so attach them only to the
    # y vars that were actually requested
    note_args=()
    for y in "${yvars[@]}"; do
        if [[ "$y" == "v(1)" ]]; then
            note_args+=(--note "v(1)=Vin")
        elif [[ "$y" == "v(2)" ]]; then
            note_args+=(--note "v(2)=$note2")
        fi
    done

    echo ">>> $(basename "$cir")  ->  $out"
    "$PYTHON" "$SIM_SCRIPT" "$cir" \
        -x "$xvar" \
        -y "${yvars[@]}" \
        -t "$base" \
        -o "$out" \
        --no-clip-levels \
        "${note_args[@]}"
done

echo
echo "Done. Plots written to $GRAPHS_DIR"
