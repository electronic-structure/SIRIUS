import matplotlib.pyplot
import scipy
import colorsys
import json
import sys
import os
import copy

matplotlib.pyplot.figure(1, figsize=(18, 8))

fin = open(sys.argv[1], "r")
jin = json.load(fin)
fin.close()

legends = []
for p in jin["plot"]:
    matplotlib.pyplot.plot(p["values"], "o-", linewidth=2.0)
    legends.append("z = " + str(p["z"]))
    
matplotlib.pyplot.legend(legends)
#matplotlib.pyplot.xticks(range(30), jin["labels"], rotation=90)
xlabels = []
xlabels_loc = []
j = 0
for n in range(1, 15):
    xlabels.append("n=" + str(n))
    xlabels_loc.append(j)
    j += n

matplotlib.pyplot.xticks(xlabels_loc, xlabels, rotation=90)

matplotlib.pyplot.yscale("log")
matplotlib.pyplot.ylabel("Relative error")
matplotlib.pyplot.grid(which="major", axis="both")

#ax = matplotlib.pyplot.axis()
#matplotlib.pyplot.axis([-1, ax[1], ax[2], ax[3]])

fname = os.path.splitext(os.path.basename(sys.argv[1]))[0]
matplotlib.pyplot.savefig(fname+".pdf", format="pdf")

