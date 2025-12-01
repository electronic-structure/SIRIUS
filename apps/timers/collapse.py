#!/bin/env python

import json
import numpy as np
import copy
import sys

data = json.load(open(sys.argv[1], "r"))


def getentry(path, data):
    total_time = np.sum(data["timings"])
    if len(data["sub-timings"]) == 0:
        return [(path, total_time)]

    sub_timings = []
    for label in data["sub-timings"]:
        sub_timings += getentry(
            copy.deepcopy(path) + [label], data["sub-timings"][label]
        )
    total_sub_timings = np.sum([x[1] for x in sub_timings])
    own_time = total_time - total_sub_timings
    if own_time < 0:
        raise Exception('something went wrong')
    return [(path, own_time)] + sub_timings


for path, t in getentry(["SCF Loop"], data):
    print(";".join(path) + " " + f"{t:10.10f}")
