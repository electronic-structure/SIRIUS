import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import colorsys

with open("timer_ornl.json", "r") as fin:
    timers = json.load(fin)['flat']

# list of timers or groups of timers to show
timers_to_show = [#'+global_timer',
                  'sirius::Band::solve_for_kset',
                  'sirius::Band::diag_pseudo_potential_davidson',

                  {'inside_davidson' : {'sirius::Band::diag_pseudo_potential_davidson|alloc',
                                        'sirius::Band::diag_pseudo_potential_davidson|evp',
                                        'sirius::Band::diag_pseudo_potential_davidson|iter',
                                        'sirius::Band::diag_pseudo_potential_davidson|update_phi',
                                        'sirius::Band::diag_pseudo_potential_davidson|wf',
                                        'sirius::Band::get_h_diag',
                                        'sirius::Band::get_o_diag'}
                 },
                 {'density_and_potential' : {
                    'sirius::Density::generate_valence',
                    'qe|mix',
                    'qe|veff'}
                 }
                 ]

# plain names of timers
timers_list = []
y_ticks_label = []

for e in timers_to_show:
    # if this is a dictionary
    if type(e) == type({}):
        # loop over keys of dictionary
        for t in e:
            y_ticks_label.append(t)
            # loop over values for the given key
            for k in e[t]:
                timers_list.append(k)
    else: # this is a plain name
        timers_list.append(e)
        y_ticks_label.append(e)

# get the maximum time
max_time = 0
for t in timers_list:
    if 'sequence' not in timers[t]:
        raise RuntimeError("time sequence is not avalibale for timer %s"%(t))
    if len(timers[t]['sequence']) % 2 != 0:
        raise RuntimeError("number of values is not even for timer %s"%(t))

    max_time = max(max_time, timers[t]['sequence'][-1])
    print("max_time=%f for timer %s"%(timers[t]['sequence'][-1], t))

print("max_time=%f"%max_time)


#print timers_list
#raise RuntimeError('stop')



# create a figure and a set of subplots
fig, ax = plt.subplots(1)
#fig.set_size_inches(max_time, num_timers)
fig.set_size_inches(130, len(timers_to_show))

plt.axis([0, max_time, 0, len(timers_to_show)])

plt.xticks(numpy.arange(0, max_time, 1.0))
plt.grid(which = "major", axis = "both")

box = ax.get_position()
ax.set_position([box.x0 + 0.1, box.y0, box.width-0.1, box.height])



y_ticks_pos = [0.5 + i for i in range(len(timers_to_show))]
plt.yticks(y_ticks_pos, y_ticks_label, rotation=0)

ypos = 0
idx_c = 0
for e in timers_to_show:

    if type(e) == type({}):
        # loop over keys of dictionary
        for t in e:
            # loop over values for the given key
            for k in e[t]:
                values = timers[k]['sequence']
                rgb = colorsys.hsv_to_rgb(idx_c / float(len(timers_list)), 0.7, 0.9)
                c = "#%X%X%X"%(rgb[0]*255, rgb[1]*255, rgb[2]*255)
    
                for i in range(len(values) / 2):
                    ax.add_patch(patches.Rectangle((values[i * 2], ypos), (values[2 * i + 1] - values[2 * i]), 0.9, linewidth=0.1, edgecolor='black', facecolor=c))

                idx_c = idx_c + 1

    else:
        values = timers[e]['sequence']
        rgb = colorsys.hsv_to_rgb(idx_c / float(len(timers_list)), 0.7, 0.9)
        c = "#%X%X%X"%(rgb[0]*255, rgb[1]*255, rgb[2]*255)
    
        for i in range(len(values) / 2):
            ax.add_patch(patches.Rectangle((values[i * 2], ypos), (values[2 * i + 1] - values[2 * i]), 0.9, linewidth=0.1, edgecolor='black', facecolor=c))
            #ax.text(values[2 * i] + 0.01, ypos + 0.45, '%.3f'%(values[2 * i + 1] - values[2 * i]))
        idx_c = idx_c + 1

    ypos = ypos + 1

fig.savefig('rect.pdf')


