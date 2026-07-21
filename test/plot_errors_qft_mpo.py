#!/usr/bin/env python3
"""
Load error data files from test/output/ and plot rms|MPO-F| and relative rms|MPO-F|
as a function of chi. All r values and all types are overlaid on the same figure.

Usage:
    python plot_errors.py [--dir test/output] [--save plot.png]
"""

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def parse_filename(fname):
    """
    Parse a filename like:
        error_chi_max40_r30_complex_double.txt
    Returns dict with keys: chi_max, r, type_name
    """
    base = os.path.basename(fname).replace(".txt", "")
    m = re.match(r"error_chi_max(\d+)_r(\d+)_(.+)", base)
    if not m:
        return None
    return {
        "chi_max": int(m.group(1)),
        "r": int(m.group(2)),
        "type_name": m.group(3),
    }


def load_file(fname):
    """Load a data file, return (chi, errors) where errors is 2D array."""
    data = np.loadtxt(fname, comments="#")
    chi = data[:, 0].astype(int)
    errors = data[:, 1:]  # columns: max|MPO-F|, rms|MPO-F|, rms|F|, relative_rms
    return chi, errors


def plot_all(data_dir, save_path=None):
    """Find all error files and plot them."""
    pattern = os.path.join(data_dir, "error_chi_max*_r*_*.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching pattern: {pattern}")
        sys.exit(1)

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    # Collect all entries: list of (chi_max, type_name, r, chi, rms_err, rel_rms)
    entries = []
    chi_max_set = set()
    for f in files:
        meta = parse_filename(f)
        if meta is None:
            print(f"WARNING: Could not parse filename: {f}")
            continue
        chi, errors = load_file(f)
        chi_max_set.add(meta["chi_max"])
        entries.append((meta["chi_max"], meta["type_name"], meta["r"],
                        chi, errors[:, 1], errors[:, 3]))

    # One figure per chi_max, all (type, r) overlaid
    for chi_max in sorted(chi_max_set):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"chi_max = {chi_max}", fontsize=14)

        # Filter entries for this chi_max and sort for consistent legend
        cur = [e for e in entries if e[0] == chi_max]
        cur.sort(key=lambda e: (e[2], e[1]))  # sort by r, then type_name

        for _, type_name, r_val, chi, rms_err, rel_rms in cur:
            label = f"{type_name}, r={r_val}"
            ax1.plot(chi, rms_err, "o-", markersize=3, label=label)
            ax2.plot(chi, rel_rms, "o-", markersize=3, label=label)

        ax1.set_xlabel("chi")
        ax1.set_ylabel("rms |MPO - F|")
        ax1.set_title("rms |MPO - F|")
        ax1.set_yscale("log")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("chi")
        ax2.set_ylabel("relative rms |MPO - F|")
        ax2.set_title("relative rms |MPO - F|")
        ax2.set_yscale("log")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            base, ext = os.path.splitext(save_path)
            group_save = f"{base}_chi_max{chi_max}{ext}"
            fig.savefig(group_save, dpi=150)
            print(f"Saved: {group_save}")

    if not save_path:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot QFT MPO error data")
    parser.add_argument("--dir", default="test/output",
                        help="Directory containing error data files")
    parser.add_argument("--save", default=None,
                        help="Save plots to file instead of showing (e.g., plot.png)")
    args = parser.parse_args()

    plot_all(args.dir, save_path=args.save)
