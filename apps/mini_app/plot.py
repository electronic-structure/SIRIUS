import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

cmap = plt.get_cmap('terrain')
norm = matplotlib.colors.Normalize(0, vmax=20.4)
#norm = LogNorm(vmin=0.001, vmax=5)

f = h5py.File("rho.h5", "r")
d = f["/rho"]


#plt.plot(d[0], "-", linewidth = 1.0)
#plt.axis([0, 100, 0, 10])

plt.axis('off')
#plt.imshow(d, cmap = cmap, norm = norm, origin = 'lower', aspect=1.73, interpolation="bicubic")
plt.imshow(d, cmap = cmap, norm = norm, origin = 'lower', aspect=1, interpolation="bicubic")
#plt.colorbar()
#plt.contour(d, [0.016], colors="#000000")



plt.savefig("rho.pdf", format="pdf", transparent=True)
