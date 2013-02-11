import matplotlib.pyplot
import scipy
import colorsys
import json
import sys
import os
import copy

matplotlib.pyplot.figure(1, figsize=(8, 12))

fin = open(sys.argv[1], "r")
jin = json.load(fin)
fin.close()

energy_scale = 27.21138505

for p in jin["plot"]:
    yval = [(x - jin["Ef"]) * energy_scale for x in p["yvalues"]]
    matplotlib.pyplot.plot(jin["xaxis"], yval, "-", color = "black", linewidth = 1.0)

# Efermi    
#matplotlib.pyplot.plot([jin["xaxis"][0], jin["xaxis"][-1]], [jin["Ef"] * energy_scale, jin["Ef"] * energy_scale], 
#                       "--", color = "black", linewidth = 1.0)
matplotlib.pyplot.plot([jin["xaxis"][0], jin["xaxis"][-1]], [0 ,0], 
                       "--", color = "black", linewidth = 1.0)

matplotlib.pyplot.xticks(jin["xaxis_ticks"], jin["xaxis_tick_labels"], rotation=90)
 
# matplotlib.pyplot.yscale("log")
matplotlib.pyplot.ylabel("Energy (Ha)")
matplotlib.pyplot.grid(which = "major", axis = "both")

# one way to setup limits
#ax = matplotlib.pyplot.axis()
#matplotlib.pyplot.axis([0, jin["xaxis"][-1], -0.5, 1])

# second way to setup limits
matplotlib.pyplot.xlim(0, jin["xaxis"][-1]) 
matplotlib.pyplot.ylim(-8, 15) 

fname = os.path.splitext(os.path.basename(sys.argv[1]))[0]
matplotlib.pyplot.savefig(fname+".pdf", format="pdf")

