import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import colorsys

with open("timers.json", "r") as fin:
    timers = json.load(fin)['flat']

timers_to_show = ['+global_timer',
                  'sirius::Band::solve_for_kset',
                  'sirius::Density::generate',
                  'sirius::Potential::generate',
                  'sirius::Broyden1::mix',
                  'sirius::DFT_ground_state::symmetrize']

num_timers = 0
max_time = 0
for t in timers_to_show:
    if 'sequence' not in timers[t]:
        raise RuntimeError("time sequence is not avalibale for timer %s"%(t))
    if len(timers[t]['sequence']) % 2 != 0:
        raise RuntimeError("number of values is not even for timer %s"%(t))

    num_timers = num_timers + 1
    max_time = max(max_time, timers[t]['sequence'][-1])
    print("max_time=%f for timer %s"%(timers[t]['sequence'][-1], t))

print("max_time=%f"%max_time)

# create a figure and a set of subplots
fig, ax = plt.subplots(1)
#fig.set_size_inches(max_time, num_timers)
fig.set_size_inches(130, num_timers)

plt.axis([0, max_time, 0, num_timers])

box = ax.get_position()
ax.set_position([box.x0 + 0.1, box.y0, box.width-0.1, box.height])



y_ticks_pos = [0.5 + i for i in range(num_timers)]
y_ticks_label = [t for t in timers_to_show]
plt.yticks(y_ticks_pos, y_ticks_label, rotation=0)

ypos = 0
for t in timers_to_show:
    values = timers[t]['sequence']
    rgb = colorsys.hsv_to_rgb(ypos / float(num_timers), 0.7, 0.9)
    c = "#%X%X%X"%(rgb[0]*255, rgb[1]*255, rgb[2]*255)
    
    for i in range(len(values) / 2):
        ax.add_patch(patches.Rectangle((values[i * 2], ypos), (values[2 * i + 1] - values[2 * i]), 0.9, linewidth=0.1, edgecolor='black', facecolor=c))

    ypos = ypos + 1

fig.savefig('rect.pdf')


