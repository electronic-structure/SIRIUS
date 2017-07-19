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
z = jin['pseudo_potential']['header']['z_valence']

vloc = jin['pseudo_potential']['local_potential']
ir0=0
for i in range(len(rgrid)):
    vloc[i] = (vloc[i] * rgrid[i] + z) * rgrid[i]
    if rgrid[i] < 10: ir0 = i

#for b in jin['pseudo_potential']['beta_projectors']:
#for b in jin['pseudo_potential']['atomic_wave_functions']:
#    plt.plot(rgrid[0:len(b['radial_function'])], b['radial_function'], "-", linewidth = 2.0)

#plt.plot(rgrid, jin['pseudo_potential']['core_charge_density'], '-', linewidth=2.0)
#plt.plot(rgrid, jin['pseudo_potential']['total_charge_density'], '-', linewidth=2.0)
plt.plot(rgrid[ir0:], vloc[ir0:], '-', linewidth=2.0)
plt.grid(which = "major", axis = "both")
plt.xlim(xmax=rgrid[-1])

fname = os.path.splitext(os.path.basename(sys.argv[1]))[0]
plt.savefig(fname+".pdf", format="pdf")


