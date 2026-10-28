#!/usr/bin/env python3
"""
NGSpice simulation and plotting script.
Runs a .cir file through ngspice and plots specified parameters.
Supports custom expressions like v(1)-v(2), v(out)/v(in), 20*log10(abs(v(out))), etc.
"""

import subprocess
import sys
import os
import re
import tempfile
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def run_ngspice(cir_file, raw_file=None, save_vars=None):
    """Run ngspice on a .cir file and save raw output."""
    if not os.path.exists(cir_file):
        raise FileNotFoundError(f"Circuit file not found: {cir_file}")

    if raw_file is None:
        raw_file = tempfile.mktemp(suffix='.raw')

    # If save_vars is specified, we need to inject a .save line into the netlist
    if save_vars:
        with open(cir_file, 'r') as f:
            cir_content = f.read()
        
        # Check if .save already exists
        if '.save' not in cir_content.lower():
            # Insert .save before .end or at the end
            save_line = f".save {' '.join(save_vars)}"
            if '.end' in cir_content.lower():
                cir_content = re.sub(r'\.end', save_line + '\n.end', cir_content, flags=re.IGNORECASE)
            else:
                cir_content += f'\n{save_line}\n'
            
            # Write to temp file
            temp_cir = tempfile.mktemp(suffix='.cir')
            with open(temp_cir, 'w') as f:
                f.write(cir_content)
            cir_file = temp_cir

    cmd = ["ngspice", "-b", "-r", raw_file, cir_file]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        if os.path.exists(raw_file):
            os.unlink(raw_file)
        raise RuntimeError(f"NGSpice failed:\n{result.stderr}")

    return raw_file


def parse_raw_file(raw_file):
    """Parse ngspice binary raw file format."""
    with open(raw_file, 'rb') as f:
        content = f.read()

    # Find the "Binary:" marker (case insensitive)
    binary_idx = content.lower().find(b'binary:')
    if binary_idx == -1:
        raise ValueError("Could not find 'Binary:' marker in raw file")

    # Find the newline after "Binary:"
    newline_idx = content.find(b'\n', binary_idx)
    if newline_idx == -1:
        raise ValueError("Malformed raw file: no newline after 'Binary:'")

    text_part = content[:binary_idx]
    binary_data = content[newline_idx + 1:]

    text = text_part.decode('ascii', errors='ignore')

    lines = text.split('\n')
    plot_name = None
    variables = []
    num_vars = 0
    num_points = 0
    flags = ''

    for line in lines:
        line = line.strip()
        if line.startswith('Plotname:'):
            plot_name = line.split(':', 1)[1].strip()
        elif line.startswith('Flags:'):
            flags = line.split(':', 1)[1].strip()
        elif line.startswith('No. Variables:'):
            num_vars = int(line.split(':', 1)[1].strip())
        elif line.startswith('No. Points:'):
            num_points = int(line.split(':', 1)[1].strip())
        elif re.match(r'^\s*\d+\s', line):
            parts = line.split()
            if len(parts) >= 3:
                var_name = parts[1]
                variables.append(var_name)

    is_complex = 'complex' in flags.lower()
    bytes_per_value = 8
    if is_complex:
        bytes_per_value = 16

    expected_size = num_points * num_vars * bytes_per_value
    if len(binary_data) < expected_size:
        print(f"Warning: Binary data size ({len(binary_data)}) smaller than expected ({expected_size})")

    data = {}
    if plot_name:
        data[plot_name] = {var: [] for var in variables}

        if is_complex:
            # Complex data: pairs of doubles (real, imag)
            for point_idx in range(num_points):
                for var_idx, var in enumerate(variables):
                    offset = (point_idx * num_vars + var_idx) * 16
                    if offset + 16 <= len(binary_data):
                        real, imag = struct.unpack('dd', binary_data[offset:offset+16])
                        data[plot_name][var].append(complex(real, imag))
        else:
            # Real data: single doubles
            for point_idx in range(num_points):
                for var_idx, var in enumerate(variables):
                    offset = (point_idx * num_vars + var_idx) * 8
                    if offset + 8 <= len(binary_data):
                        val = struct.unpack('d', binary_data[offset:offset+8])[0]
                        data[plot_name][var].append(val)

    # For complex data, also add magnitude and phase as derived variables
    if is_complex and plot_name:
        plot_data = data[plot_name]
        for var in list(plot_data.keys()):
            values = plot_data[var]
            if values and isinstance(values[0], complex):
                arr = np.array(values)
                # For frequency and other known real variables, take real part
                if var.lower() in ('frequency', 'time'):
                    plot_data[var] = np.real(arr).tolist()
                else:
                    plot_data[f"mag({var})"] = np.abs(arr).tolist()
                    plot_data[f"phase({var})"] = np.angle(arr, deg=True).tolist()
                    plot_data[f"real({var})"] = np.real(arr).tolist()
                    plot_data[f"imag({var})"] = np.imag(arr).tolist()

    return data


def evaluate_expression(expr, variables_dict):
    """
    Evaluate a custom expression using available variables.
    Supports basic arithmetic: +, -, *, /, **, ()
    Variables can be referenced as v(1), v(2), i(v1), mag(v(out)), phase(v(out)), etc.
    """
    # Create a safe namespace with numpy functions
    namespace = {
        'np': np,
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'exp': np.exp, 'log': np.log, 'log10': np.log10,
        'sqrt': np.sqrt, 'abs': np.abs,
        'pi': np.pi, 'e': np.e,
        'mag': np.abs,
        'phase': lambda x: np.angle(x, deg=True),
        'real': np.real,
        'imag': np.imag,
    }
    
    # Add all variables to namespace with sanitized names
    # Map variable names like "v(1)" to "v_1", "v(2)" to "v_2", etc.
    var_mapping = {}
    for name, values in variables_dict.items():
        # Sanitize variable name for Python identifier
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        if not safe_name or safe_name[0].isdigit():
            safe_name = 'var_' + safe_name
        namespace[safe_name] = np.array(values)
        var_mapping[name] = safe_name
    
    # Replace variable names in expression with safe names
    # Sort by length descending to handle nested names correctly
    sorted_vars = sorted(var_mapping.keys(), key=len, reverse=True)
    safe_expr = expr
    for var in sorted_vars:
        safe_expr = safe_expr.replace(var, var_mapping[var])
    
    # For safety, we'll use eval with a restricted namespace
    # Use namespace as both globals and locals to allow operators
    try:
        result = eval(safe_expr, namespace, namespace)
        return np.array(result)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{expr}': {e}")


def compute_expressions(data, plot_name, y_vars):
    """Compute custom expressions and add them to the data."""
    if plot_name not in data:
        return data
    
    plot_data = data[plot_name]
    variables_dict = {k: np.array(v) for k, v in plot_data.items()}
    
    for expr in y_vars:
        # Check if this is an expression (contains operators not in variable names)
        is_expression = (any(op in expr for op in ['+', '-', '*', '/', '**']) 
                        and expr not in plot_data)
        
        if is_expression and expr not in plot_data:
            try:
                result = evaluate_expression(expr, variables_dict)
                plot_data[expr] = result.tolist()
                print(f"Computed expression: {expr} ({len(result)} points)")
            except Exception as e:
                print(f"Warning: Could not compute expression '{expr}': {e}")
    
    return data


def plot_data(data, x_var='time', y_vars=None, plot_name=None, title=None, save_path=None):
    """Plot simulation data using matplotlib."""
    if plot_name and plot_name in data:
        plot_data = data[plot_name]
    elif data:
        plot_name = list(data.keys())[0]
        plot_data = data[plot_name]
    else:
        raise ValueError("No data to plot")

    if x_var not in plot_data:
        available = list(plot_data.keys())
        raise ValueError(f"X variable '{x_var}' not found. Available: {available}")

    if y_vars is None:
        y_vars = [v for v in plot_data.keys() if v != x_var]

    x = np.array(plot_data[x_var])

    fig, axes = plt.subplots(len(y_vars), 1, figsize=(10, 4 * len(y_vars)), squeeze=False)
    axes = axes.flatten()

    for idx, y_var in enumerate(y_vars):
        if y_var not in plot_data:
            print(f"Warning: '{y_var}' not found in data, skipping")
            continue

        y = np.array(plot_data[y_var])
        ax = axes[idx]
        ax.plot(x, y, 'b-', linewidth=1)
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_title(f"{y_var} vs {x_var}")
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run ngspice simulation and plot results")
    parser.add_argument("cir_file", help="Path to .cir circuit file")
    parser.add_argument("-x", "--xvar", default="time", help="X-axis variable (default: time)")
    parser.add_argument("-y", "--yvars", nargs="+", help="Y-axis variables or expressions to plot (e.g., 'v(1)-v(2)', 'v(out)/v(in)', '20*log10(mag(v(out)))')")
    parser.add_argument("-p", "--plot", help="Specific plot name to use")
    parser.add_argument("-t", "--title", help="Plot title")
    parser.add_argument("-o", "--output", help="Save plot to file")
    parser.add_argument("--raw", help="Save raw output to file (default: temp file)")
    parser.add_argument("--list", action="store_true", help="List available variables and exit")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw file after simulation")
    parser.add_argument("--save-all", action="store_true", help="Save all available variables (uses .save in netlist)")

    args = parser.parse_args()

    print(f"Running simulation on {args.cir_file}...")

    raw_file = args.raw
    keep_raw = args.keep_raw or bool(args.raw)

    try:
        raw_path = run_ngspice(args.cir_file, raw_file, None)
        data = parse_raw_file(raw_path)

        if args.list:
            for plot_name, vars_dict in data.items():
                print(f"\nPlot: {plot_name}")
                for var in vars_dict:
                    print(f"  {var}: {len(vars_dict[var])} points")
            return

        # Compute expressions if any
        if args.yvars:
            plot_name = args.plot or list(data.keys())[0]
            data = compute_expressions(data, plot_name, args.yvars)

        plot_data(data, x_var=args.xvar, y_vars=args.yvars, plot_name=args.plot,
                  title=args.title, save_path=args.output)

    finally:
        if not keep_raw and 'raw_path' in locals() and raw_path and os.path.exists(raw_path):
            os.unlink(raw_path)


if __name__ == "__main__":
    main()