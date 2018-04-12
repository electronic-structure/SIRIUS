from .bands import *
from .helpers import *

from pathlib import Path
import sys
import json

import pkgutil
from pkg_resources import resource_filename
#path_dict = json.load(open('includes/path_dict.json'))
#open(resource_filename('sirius', 'includes/path_dict.json'), 'rb')
data = pkgutil.get_data("sirius", "includes/path_dict.json")
path_dict = json.loads(data.decode())

#path_dict = json.load(pkgutil.get_data("sirius", "includes/path_dict.json"))

bindings_path = path_dict["path"]


if bindings_path == "":
    paths = []
    for file_path in Path('/Users/').glob('**/py_sirius.cpython-36m-darwin.so'):
        paths.append(file_path)
    temp = list(str(paths[0]))
    for i in range(32):
        temp.pop()
    import_path = "".join(temp)
    #write to path_dict
    new_path = {"path": import_path}
    with open(resource_filename('sirius', 'includes/path_dict.json'), 'w') as outfile:
        json.dump(new_path, outfile)
    sys.path.insert(1, import_path)
    from py_sirius import *

else:
    sys.path.insert(1, bindings_path)
    from py_sirius import *
