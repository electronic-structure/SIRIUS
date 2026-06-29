#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt


def plot_json_to_pdf(json_file: str, output_pdf: str) -> None:
    # Expected JSON structure:
    # {
    #   "t":       [0, 1, 2, ...],
    #   "val_abs": [ ... ],
    #   "val_re":  [ ... ],
    #   "val_im":  [ ... ]
    # }

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    t = data["t"]
    val_abs = data["val_abs"]
    val_re = data["val_re"]
    val_im = data["val_im"]

    plt.figure(figsize=(8, 5))

    plt.plot(t, val_abs, label="abs")
    plt.plot(t, val_re, label="real")
    plt.plot(t, val_im, label="imag")

    plt.xlabel("t")
    plt.ylabel("value")
    plt.title("Values vs time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()


if __name__ == "__main__":
    plot_json_to_pdf("psi_r.json", "output.pdf")
