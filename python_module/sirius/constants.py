from .py_sirius import spin_range  # type: ignore

# collinear case
spin_up = spin_range(0, 1)
spin_dn = spin_range(1, 2)

# non-collinear case
spin_ud = spin_range(0, 2)
