#!/usr/bin/env python3

import json
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot one radial function u(r) from SIRIUS JSON output.")
    parser.add_argument("input", help="Input JSON file")
    parser.add_argument("--idxrf", type=int, required=True, help="Radial-function index to plot")
    parser.add_argument("-o", "--output", default="radial_function_u.pdf", help="Output PDF file")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    r = data["x"]

    matches = [rf for rf in data["radial_functions"] if rf.get("idxrf") == args.idxrf]

    if not matches:
        raise SystemExit(f"idxrf={args.idxrf} not found")

    fig, ax = plt.subplots(figsize=(8, 5))

    for rf in matches:
        u = rf["u"]

        label = f"idxrf={rf.get('idxrf')}, l={rf.get('l')}, order={rf.get('order')}"
        ax.plot(r, u, label=label)

    ax.set_xlabel("r")
    ax.set_ylabel("u(r)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
