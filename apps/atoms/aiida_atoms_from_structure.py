from aiida.orm.querytool import QueryTool
from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm import Group
import os, sys, stat

structure_grp = 'test_structures' # 'Cottenier_structures',

struct_q = QueryTool()
struct_q.set_class(StructureData)
struct_q.set_group(structure_grp, exclude=False)

symbols = []
for node in struct_q.run_query():
    for s in node.get_symbols_set():
        if not s in symbols: symbols.append(s) 

fout = open("run.x", "w")
for s in symbols:
    t = sys.argv[1] if len(sys.argv) >= 2 else " --type=lo1+lo2+lo3+LO1"
    fout.write("./atom --symbol=" + s + t + "\n");
fout.close()
os.chmod("run.x", os.stat("run.x").st_mode | stat.S_IEXEC)
