import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import colorsys

with open("time_seq.json", "r") as fin:
    timers = json.load(fin)

num_timers = 0
max_time = 0
for t in timers:
    if len(timers[t]) % 2 != 0:
        raise RuntimeError("number of values is not even")

    num_timers = num_timers + 1
    max_time = max(max_time, timers[t][-1])

# create a figure and a set of subplots
fig, ax = plt.subplots(1)
fig.set_size_inches(max_time, num_timers)

plt.axis([0, max_time, 0, num_timers])

box = ax.get_position()
ax.set_position([box.x0 + 0.1, box.y0, box.width-0.1, box.height])



y_ticks_pos = [0.5 + i for i in range(num_timers)]
y_ticks_label = [t for t in timers]
plt.yticks(y_ticks_pos, y_ticks_label, rotation=0)

ypos = 0
for t in timers:
    values = timers[t]
    
    rgb = colorsys.hsv_to_rgb(ypos / float(num_timers), 0.7, 0.9)
    c = "#%X%X%X"%(rgb[0]*255, rgb[1]*255, rgb[2]*255)
    
    for i in range(len(values) / 2):
        ax.add_patch(patches.Rectangle((values[i * 2], ypos), (values[2 * i + 1] - values[2 * i]), 0.9, linewidth=0.1, edgecolor='black', facecolor=c))

    ypos = ypos + 1

fig.savefig('rect.pdf')


