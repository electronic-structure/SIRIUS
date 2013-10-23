import matplotlib.pyplot
import scipy
import colorsys
import json
import sys
import os
import copy

matplotlib.pyplot.figure(1, figsize=(14,14))

fname = os.path.splitext(os.path.basename(sys.argv[1]))[0]

fin = open(sys.argv[1], "r")
jin = json.load(fin)
fin.close()

# 4 timer values are: total, min, max, average
timer_groups = [
[
    "sirius::Global::generate_radial_functions",
    "sirius::Global::generate_radial_integrals", 
    "sirius::K_set::find_eigen_states",
    "sirius::Density::generate",
    "sirius::Potential::generate_effective_potential",
    "exciting::sym_rho_mag",
    "exciting::mixer"
],
[
    "sirius::Band::set_fv_h_o",
    "sirius::Band::solve_fv_evp",
    "sirius::K_point::generate_fv_states",
    "sirius::Band::solve_sv",
    "sirius::K_point::generate_spinor_wave_functions"
],
[
    "sirius::Potential::poisson",
    "sirius::Potential::xc"
],
[
    "sirius::Reciprocal_lattice::init",
    "sirius::Step_function::init",
    "sirius::Unit_cell::get_symmetry",
    "sirius::Unit_cell::find_nearest_neighbours",
    "sirius::K_point::initialize",
    "sirius::Potential::Potential",
    "sirius::Atom_type::solve_free_atom"
]
]

for itg in range(len(timer_groups)):

    timer_names = []
    timer_values = []
    total_time = 0.0
    for timer_name in timer_groups[itg]:
        
        if timer_name in jin["timers"]:
            
            tname = timer_name
            
            # effective potential is generated once before the scf loop
            # the first timer is reported in percentage
            if itg == 0: 
                if timer_name == "sirius::Potential::generate_effective_potential":
                    # (total - average) of effective potential / total of iterations
                    t = (jin["timers"][timer_name][0] - jin["timers"][timer_name][3]) / jin["timers"]["exciting::iteration"][0]
                else:
                    t = jin["timers"][timer_name][0] / jin["timers"]["exciting::iteration"][0]
                t = t * 100
                # show average time in legend
                timer_names.append(tname + " (%6.2f%%, %6.2f sec./call)"%(t, jin["timers"][timer_name][3]))
            # show total time for intialization routines
            elif itg == 3:
                t = jin["timers"][timer_name][0]
                timer_names.append(tname + " (%6.2f sec.)"%t)
            # show average time
            else:
                t = jin["timers"][timer_name][3]
                timer_names.append(tname + " (%6.2f sec./call)"%t)
            
            timer_values.append(t)
            total_time += t
        
    print "total time for timer group ", itg, " ", total_time
    
    plot = matplotlib.pyplot.subplot("41%i"%(itg+1))
    box = plot.get_position()
    plot.set_position([box.x0, box.y0, box.width * 0.1, box.height])
    box = plot.get_position()
    
    ytics = [0]
    for i in range(len(timer_values)):
        ytics.append(ytics[i] + timer_values[i])
    
    plots = []
    for i in range(len(timer_values)):
        rgb = colorsys.hsv_to_rgb(i / float(len(timer_values)), 0.75, 0.95)
        c = "#%X%X%X"%(rgb[0]*255, rgb[1]*255, rgb[2]*255)
        plots.append(matplotlib.pyplot.bar(0, timer_values[i], 2, bottom=ytics[i], color=c))
    
    matplotlib.pyplot.xticks([], ())
    matplotlib.pyplot.yticks(ytics)
    matplotlib.pyplot.ylim([0, ytics[len(ytics)-1]])
    
    matplotlib.pyplot.legend(plots[::-1], timer_names[::-1], bbox_to_anchor=(1.2, 1), loc=2)

matplotlib.pyplot.savefig(fname+".pdf", format="pdf")

