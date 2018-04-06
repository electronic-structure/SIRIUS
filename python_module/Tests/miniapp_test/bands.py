import sys
sys.path.insert(0, '../../')

import matplotlib.pyplot as plt
import scipy
import colorsys
import json
import sys
import os
import sirius
import copy

def plotter(jin1, label1, jin2 = [], label2 = "", both = False, filename = "band_plot.pdf"):
    plt.figure(1, figsize=(8, 12))

    emin = -0.5
    emax = 1

    energy_scale = 27.21138505

    bands = jin1["bands"]
    x_axis = jin1["header"]["x_axis"]
    x_ticks = jin1["header"]["x_ticks"]

    num_bands = jin1["header"]["num_bands"]
    #print("num_bands1=", num_bands)



    bnd_e = [e["values"][0] for e in bands]
    plt.plot(x_axis, bnd_e, "-", color = "black", linewidth = 1.0, label = label1)
    for i in range(num_bands-1):
        bnd_e = [e["values"][i+1] for e in bands]
        plt.plot(x_axis, bnd_e, "-", color = "black", linewidth = 1.0)

    if both == True:
        bands2 = jin2["bands"]
        x_axis = jin2["header"]["x_axis"]
        x_ticks = jin2["header"]["x_ticks"]

        num_bands2 = jin2["header"]["num_bands"]
        #print("num_bands2=", num_bands2)

        bnd_e = [e["values"][0] for e in bands2]
        plt.plot(x_axis, bnd_e, "-.", color = "red", linewidth = 1.0, label = label2)
        for i in range(num_bands2-1):
            bnd_e = [e["values"][i+1] for e in bands2]
            plt.plot(x_axis, bnd_e, "-.", color = "red", linewidth = 1.0)

    x_ticks_pos = [e["x"] for e in x_ticks]

    x_ticks_label = [e["label"] for e in x_ticks]
    plt.legend(loc = 'best')
    plt.xticks(x_ticks_pos, x_ticks_label, rotation=0)

    plt.ylabel("Energy (Ha)")
    plt.grid(which = "major", axis = "both")

    # setup x and y limits
    ax = plt.axis()
    plt.axis([0, x_axis[-1], emin, emax])
    plt.savefig(filename, format="pdf")


def get_kpoint_path(k_point_list, rec_vec, ctx): #return list of vectors and xticks
    x_axis = []
    x_ticks = []
    vertex = []
    kpoints = []
    for i in enumerate(k_point_list):
        v = rec_vec[i[1]] # need K-points_rel == k_point_list
        vertex.append((i[1],v))

    x_axis.append(0)
    x_ticks.append((0, vertex[0][0]))
    #print(vertex[0][1])
    kpoints.append(sirius.vector3d_double(vertex[0][1]))

    t = 0
    #print("vertex_type ="  + str(type(vertex)))
    #print("vertex_length =" + str(len(vertex)))
    for i in range(len(vertex)-1):
        v0 = sirius.vector3d_double(vertex[i][1])
        v1 = sirius.vector3d_double(vertex[i+1][1])
        dv = v1 - v0
        dv_cart = ctx.unit_cell().reciprocal_lattice_vectors() * dv
        np = max(10, int(30*dv_cart.length()))
        for j in range(1, np+1):
            v = v0 + dv*(float(j)/np)
            kpoints.append(v)
            t += dv_cart.length()/np
            x_axis.append(t)
        x_ticks.append((t, vertex[i+1][0]))
    return_dict = {"k_points" : kpoints, "x_ticks" : x_ticks}
    for i in range(len(kpoints)):
        x = kpoints[i]
        #print("Type of kpoints elements is:", type(x))
        #print(x(0))
    return kpoints, x_ticks, x_axis
