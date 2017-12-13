import matplotlib.pyplot as plt
import scipy
import colorsys
import json
import sys
import os
import copy

plt.figure(1, figsize=(8, 12))

emin = -0.5
emax = 1

with open(sys.argv[1], "r") as fin:
    jin = json.load(fin)

energy_scale = 27.21138505

bands = jin["bands"]
x_axis = jin["header"]["x_axis"]
x_ticks = jin["header"]["x_ticks"]

num_bands = jin["header"]["num_bands"]
for i in range(num_bands):
    bnd_e = [e["values"][i] for e in bands]

    plt.plot(x_axis, bnd_e, "-", color = "black", linewidth = 1.0)

#for p in jin["plot"]:
#    yval = [(x - jin["Ef"]) * energy_scale for x in p["yvalues"]]
#    matplotlib.pyplot.plot(jin["xaxis"], yval, "-", color = "black", linewidth = 1.0)

## Efermi    
##matplotlib.pyplot.plot([jin["xaxis"][0], jin["xaxis"][-1]], [jin["Ef"] * energy_scale, jin["Ef"] * energy_scale], 
##                       "--", color = "black", linewidth = 1.0)
#matplotlib.pyplot.plot([jin["xaxis"][0], jin["xaxis"][-1]], [0 ,0], 
#                       "--", color = "black", linewidth = 1.0)
#
x_ticks_pos = [e["x"] for e in x_ticks]
x_ticks_label = [e["label"] for e in x_ticks]
plt.xticks(x_ticks_pos, x_ticks_label, rotation=0)
# 
## matplotlib.pyplot.yscale("log")
plt.ylabel("Energy (Ha)")
plt.grid(which = "major", axis = "both")
# setup x and y limits
ax = plt.axis()
plt.axis([0, x_axis[-1], emin, emax])

# second way to setup limits
#matplotlib.pyplot.xlim(0, jin["xaxis"][-1]) 
#matplotlib.pyplot.ylim(-8, 15) 

fname = os.path.splitext(os.path.basename(sys.argv[1]))[0]
plt.savefig(fname+".pdf", format="pdf")

