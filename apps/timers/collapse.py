#!/bin/env python

import json
import numpy as np
import copy
import sys


data = json.load(open(sys.argv[1], "r"))


def getentry(path, data):
    if len(data["sub-timings"]) == 0:
        yield path, np.sum(data["timings"])
    for label in data["sub-timings"]:
        yield from getentry(copy.deepcopy(path) + [label], data["sub-timings"][label])


for path, t in getentry(["sirius"], data["sirius"]):
    print(";".join(path) + " " + f"{t:10.10f}")
