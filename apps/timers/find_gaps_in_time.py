import json

with open('timers.json') as json_file:
    timers = json.load(json_file)

timeline = []

for name in timers["sirius"]["sub-timings"]:
    l = len(timers["sirius"]["sub-timings"][name]["start-times"])
    for i in range(l):
        t0 = timers["sirius"]["sub-timings"][name]["start-times"][i]
        t1 = timers["sirius"]["sub-timings"][name]["timings"][i]
        timeline.append((t0, name, "start"))
        timeline.append((t0 + t1, name, "stop"))

sorted_timeline = sorted(timeline)
for i in range(len(sorted_timeline) - 1):
    a = sorted_timeline[i]
    if a[2] == "stop":
        b = sorted_timeline[i + 1]
        if b[2] != "start":
            print(b)
            print(a)
            raise Exception("something is wrong")
        if b[0] - a[0] > 1.0:
            print("\n")
            print(a)
            print(b)

