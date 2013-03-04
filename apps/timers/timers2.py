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

timer_groups = [
[
    "sirius::Global::generate_radial_functions",
    "sirius::Global::generate_radial_integrals", 
    "sirius::kpoint::find_eigen_states",
    "sirius::Density::generate",
    "sirius::Potential::generate_effective_potential",
    "elk::symmetrization",
    "elk::mixer"
],
[
    "sirius::kpoint::set_fv_h_o",
    "sirius::kpoint::generate_fv_states:genevp",
    "sirius::kpoint::generate_fv_states:wf",
    "sirius::Band::solve_sv",
    "sirius::kpoint::generate_spinor_wave_functions"
],
[
    "sirius::Potential::poisson",
    "sirius::Potential::xc"
],
[
    "sirius::ReciprocalLattice::init",
    "sirius::StepFunction::init",
    "sirius::UnitCell::get_symmetry",
    "sirius::UnitCell::find_nearest_neighbours",
    "sirius::kpoint::initialize",
    "sirius::Potential::Potential",
    "sirius::AtomType::solve_free_atom"
]
]

for itg in range(len(timer_groups)):

    timer_names = []
    timer_values = []
    total_time = 0.0
    for timer_name in timer_groups[itg]:
        
        if timer_name in jin["timers"]:
            
            #tname = timer_name[timer_name.rfind("::")+2:]
            tname = timer_name
            
            if itg == 0: 
                if timer_name == "sirius::Potential::generate_effective_potential":
                    t = (jin["timers"][timer_name][0] - jin["timers"][timer_name][1]) / jin["timers"]["elk::iteration"][0]
                else:
                    t = jin["timers"][timer_name][0] / jin["timers"]["elk::iteration"][0]
                t = t * 100
                timer_names.append(tname + " (%6.2f%%, %6.2f sec./call)"%(t, jin["timers"][timer_name][1]))
            elif itg == 3:
                t = jin["timers"][timer_name][0]
                timer_names.append(tname + " (%6.2f sec.)"%t)
            else:
                t = jin["timers"][timer_name][1]
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

