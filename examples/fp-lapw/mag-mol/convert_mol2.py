import numpy as np
import json

def read_mol2_atoms(filename):
    atoms = []
    discarded = []
    inside_atoms = False

    with open(filename) as file:
        for line in file:
            line = line.strip()

            if line == "@<TRIPOS>ATOM":
                inside_atoms = True
                continue

            if line.startswith("@<TRIPOS>") and inside_atoms:
                break

            if not inside_atoms or not line:
                continue

            fields = line.split()

            atom = {
                "id": int(fields[0]),
                "name": fields[1],
                "position": np.array([
                    float(fields[2]),
                    float(fields[3]),
                    float(fields[4]),
                ], dtype=float),
                "type": fields[5],
                "residue_id": int(fields[6]),
                "residue_name": fields[7],
            }

            if atom["name"].endswith("?"):
                discarded.append(atom)
            else:
                atoms.append(atom)

    return atoms, discarded


atoms, discarded = read_mol2_atoms("2097929.mol2")

print(f"Retained:  {len(atoms)} atoms")
print(f"Discarded: {len(discarded)} disorder alternatives")

for atom in discarded:
    print(atom["id"], atom["name"])

# Locate Tb1 and Tb2 by their MOL2 atom names.
by_name = {atom["name"]: atom for atom in atoms}

try:
    r_tb1 = by_name["Tb1"]["position"]
    r_tb2 = by_name["Tb2"]["position"]
except KeyError:
    print("Available Tb atom names:")
    print([a["name"] for a in atoms if a["type"].startswith("Tb")])
    raise


# Tb–Tb distance and midpoint
tb_vector = r_tb2 - r_tb1
L = np.linalg.norm(tb_vector)
midpoint = 0.5 * (r_tb1 + r_tb2)

# New z-axis: Tb1 -> Tb2
ez = tb_vector / L

# Choose the Cartesian direction least parallel to ez.
# This avoids an unstable cross product.
cartesian_axes = np.eye(3)
reference = cartesian_axes[np.argmin(np.abs(cartesian_axes @ ez))]

# Construct a right-handed orthonormal basis.
ex = np.cross(reference, ez)
ex /= np.linalg.norm(ex)

ey = np.cross(ez, ex)

# Rows of R are the new basis vectors expressed in old coordinates.
R = np.vstack([ex, ey, ez])

# Translate midpoint to the origin, then rotate every atom.
for atom in atoms:
    shifted = atom["position"] - midpoint
    atom["rotated_position"] = R @ shifted


print(f"Tb–Tb distance L = {L:.10f} Å")
print("Tb1:", by_name["Tb1"]["rotated_position"])
print("Tb2:", by_name["Tb2"]["rotated_position"])

print("\nRotated coordinates [Å]:")

for atom in atoms:
    x, y, z = atom["rotated_position"]
    print(f"{atom['name']:8s} {x:16.10f} {y:16.10f} {z:16.10f}")

symbols = []
coordinates = {}
with open("molecule_rotated.xyz", "w") as file:
    file.write(f"{len(atoms)}\n")
    file.write(f"Tb-Tb distance: {L:.10f} Angstrom\n")

    for atom in atoms:
        # MOL2 types such as C.3 and C.ar become C
        symbol = atom["type"].split(".")[0]
        x, y, z = atom["rotated_position"]

        if symbol not in coordinates:
            symbols.append(symbol)
            coordinates[symbol] = []

        file.write(
            f"{symbol:2s} "
            f"{x:18.10f} {y:18.10f} {z:18.10f}\n"
        )

        coordinates[symbol].append([
            float(x),
            float(y),
            float(z),
        ])

unit_cell = {
    "atom_types": symbols,
    "atom_files": {
        symbol: f"{symbol}.json"
        for symbol in symbols
    },
    "atoms": coordinates,
    "atom_coordinate_units": "A",
    "lattice_vectors": [
       [1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
       [0.0, 0.0, 1.0],
    ],
    "lattice_vectors_scale": 20.0,
}

with open("unit_cell.json", "w", encoding="utf-8") as output:
    json.dump({"unit_cell": unit_cell}, output, indent=2)
