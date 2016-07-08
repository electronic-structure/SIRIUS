import json
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator as o

import numpy as np

jf1 = json.load(open(sys.argv[1], 'r'))
jf2 = json.load(open(sys.argv[2], 'r'))
jf3 = json.load(open(sys.argv[3], 'r'))

d = []

for key in jf1['timers']:
    t = jf3['timers'][key][0]
    if t > 5:
        d.append([sys.argv[1], key, jf1['timers'][key][0]])
        d.append([sys.argv[2], key, jf2['timers'][key][0]])
        d.append([sys.argv[3], key, jf3['timers'][key][0]])

dpoints = np.array(d)

fig = plt.figure()
ax = fig.add_subplot(111)

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.6, box.width, box.height * 0.5])

def barplot(ax, dpoints):
    '''
    Create a barchart for data across different categories with
    multiple conditions for each category.
    
    @param ax: The plotting axes from matplotlib.
    @param dpoints: The data set as an (n, 3) numpy array
    '''
    
    # Aggregate the conditions and the categories according to their
    # mean values
    conditions = [(c, np.mean(dpoints[dpoints[:,0] == c][:,2].astype(float))) 
                  for c in np.unique(dpoints[:,0])]
    categories = [(c, np.mean(dpoints[dpoints[:,1] == c][:,2].astype(float))) 
                  for c in np.unique(dpoints[:,1])]
    
    # sort the conditions, categories and data so that the bars in
    # the plot will be ordered by category and condition
    conditions = [c[0] for c in sorted(conditions, key=o.itemgetter(1))]
    categories = [c[0] for c in sorted(categories, key=o.itemgetter(1))]
    
    dpoints = np.array(sorted(dpoints, key=lambda x: categories.index(x[1])))

    # the space between each set of bars
    space = 0.3
    n = len(conditions)
    width = (1 - space) / (len(conditions))
    
    # Create a set of bars at each position
    for i,cond in enumerate(conditions):
        indeces = range(1, len(categories)+1)
        vals = dpoints[dpoints[:,0] == cond][:,2].astype(np.float)
        pos = [j - (1 - space) / 2. + i * width for j in indeces]
        ax.bar(pos, vals, width=width, label=cond, 
               color=cm.Accent(float(i) / n))
    
    # Set the x-axis tick labels to be equal to the categories
    ax.set_xticks(indeces)
    ax.set_xticklabels(categories)
    plt.setp(plt.xticks()[1], rotation=90)
    
    # Add the axis labels
    ax.set_ylabel("Time (sec.)")
    ax.set_xlabel("Timer names")
    
    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left')
        
barplot(ax, dpoints)
plt.savefig('timers.pdf', format='pdf')
plt.show()
