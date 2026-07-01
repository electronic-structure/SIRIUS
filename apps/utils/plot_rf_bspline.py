#!/usr/bin/env python3

import json
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", default="bspline_radial_functions.json")
    parser.add_argument("--u", action="store_true", help="plot u(r) instead of p(r)=r*u(r)")
    args = parser.parse_args()

    with open(args.file) as f:
        data = json.load(f)

    r = data["r"]
    key = "u" if args.u else "p"
    y = data[key]
    evals = data["eval"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, vals in enumerate(y[:2]):
        ax.plot(r, vals, label=f"{i}: E={evals[i]:.8f}")

    ax.set_xlabel("r")
    ax.set_ylabel("u(r)" if args.u else "p(r) = r u(r)")
    ax.set_title(f"B-spline radial functions, l={data.get('l')}, order={data.get('order')}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    fig.savefig("plot.pdf", bbox_inches="tight")



if __name__ == "__main__":
    main()
