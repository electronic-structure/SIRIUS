import matplotlib.pyplot
import scipy
import colorsys
import json
import sys
import os
import copy

matplotlib.pyplot.figure(1, figsize=(20, 5))

fin = open(sys.argv[1], "r")
jin = json.load(fin)
fin.close()

legends = []
for p in jin["plot"]:

    if "xaxis" in p: 
        matplotlib.pyplot.plot(p["xaxis"], p["yvalues"], "o-", linewidth = 2.0)
    else:
        matplotlib.pyplot.plot(jin["xaxis"], p["yvalues"], "o-", linewidth = 2.0)

    legends.append(p["label"])
    
matplotlib.pyplot.legend(legends)
matplotlib.pyplot.xticks(jin["xaxis_ticks"][:15], jin["xaxis_tick_labels"][:15], rotation=90)
 
matplotlib.pyplot.yscale("log")
matplotlib.pyplot.ylabel("Relative error")
matplotlib.pyplot.grid(which = "major", axis = "both")

#ax = matplotlib.pyplot.axis()
#matplotlib.pyplot.axis([-1, ax[1], ax[2], ax[3]])

fname = os.path.splitext(os.path.basename(sys.argv[1]))[0]
matplotlib.pyplot.savefig(fname+".pdf", format="pdf")

