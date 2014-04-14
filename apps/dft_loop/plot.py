import h5py
import matplotlib
import matplotlib.pyplot as plt

cmap = plt.get_cmap('rainbow')
norm = matplotlib.colors.Normalize(0, vmax=50)

f = h5py.File("rho.h5", "r")
d = f["/rho"]

plt.axis('off')
plt.imshow(d, cmap = cmap, norm = norm, origin = 'lower')
plt.colorbar()
plt.contour(d, [1, 5, 10, 20, 30, 40, 50])
plt.savefig("rho.pdf", format="pdf")
