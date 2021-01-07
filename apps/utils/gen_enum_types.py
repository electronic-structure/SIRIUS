name = "relativity"

members = ["none", "koelling_harmon", "zora", "iora", "dirac"]

print(f"enum class {name}_t")
print("{")
sep = ""
for m in members:
    print(f"{sep}    {m}", end="")
    sep = ",\n"
print("\n};")
print("\n")
print(f"inline {name}_t get_{name}_t(std::string name__)")
print("{")
print("    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);")
sep = ""
print(f"    std::map<std::string, {name}_t> const m = {{")
for m in members:
    print(f'{sep}        {{"{m.lower()}", {name}_t::{m}}}', end="")
    sep = ",\n"
print("\n    };\n")
print("    if (m.count(name__) == 0) {")
print("        std::stringstream s;")
print(f'        s << "get_{name}_t(): wrong label of the {name}_t enumerator: " << name__;')
print("        throw std::runtime_error(s.str());")
print("     }")
print("     return m.at(name__);")
print("}")

