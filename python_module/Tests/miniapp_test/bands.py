import matplotlib.pyplot as plt
import scipy
import colorsys
import json
import sys
import os
import copy

def plotter(jin1, jin2 = [], both = False):
    plt.figure(1, figsize=(8, 12))

    emin = -0.5
    emax = 1

    energy_scale = 27.21138505

    bands = jin1["bands"]
    x_axis = jin1["header"]["x_axis"]
    x_ticks = jin1["header"]["x_ticks"]

    num_bands = jin1["header"]["num_bands"]
    print("num_bands1=", num_bands)

    for i in range(num_bands):
        bnd_e = [e["values"][i] for e in bands]
        plt.plot(x_axis, bnd_e, "-", color = "black", linewidth = 1.0)

    if both == True:
        bands2 = jin2["bands"]
        x_axis = jin2["header"]["x_axis"]
        x_ticks = jin2["header"]["x_ticks"]

        num_bands2 = jin2["header"]["num_bands"]
        print("num_bands2=", num_bands2)

        for i in range(num_bands2):
            bnd_e = [e["values"][i] for e in bands2]
            plt.plot(x_axis, bnd_e, "-.", color = "red", linewidth = 1.0)

    x_ticks_pos = [e["x"] for e in x_ticks]

    x_ticks_label = [e["label"] for e in x_ticks]
    plt.xticks(x_ticks_pos, x_ticks_label, rotation=0)

    plt.ylabel("Energy (Ha)")
    plt.grid(which = "major", axis = "both")

    # setup x and y limits
    ax = plt.axis()
    plt.axis([0, x_axis[-1], emin, emax])
    plt.savefig("band_plot.pdf", format="pdf")
