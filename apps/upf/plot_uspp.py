import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import json
import os


fin = open(sys.argv[1], "r")
jin = json.load(fin)
fin.close()

rgrid = jin['pseudo_potential']['radial_grid']

#for b in jin['pseudo_potential']['beta_projectors']:
#for b in jin['pseudo_potential']['atomic_wave_functions']:
#    plt.plot(rgrid[0:len(b['radial_function'])], b['radial_function'], "-", linewidth = 2.0)

#plt.plot(rgrid, jin['pseudo_potential']['core_charge_density'], '-', linewidth=2.0)
#plt.plot(rgrid, jin['pseudo_potential']['total_charge_density'], '-', linewidth=2.0)
plt.plot(rgrid, jin['pseudo_potential']['local_potential'], '-', linewidth=2.0)
plt.grid(which = "major", axis = "both")
plt.xlim(xmax=10)

fname = os.path.splitext(os.path.basename(sys.argv[1]))[0]
plt.savefig(fname+".pdf", format="pdf")


