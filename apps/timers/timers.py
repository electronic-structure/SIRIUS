import matplotlib.pyplot
import scipy
import colorsys
import json
import sys
import os
import copy

matplotlib.pyplot.figure(1, figsize=(10,14))

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

    timer_names = copy.deepcopy(timer_groups[itg])
    timer_values = []
    total_time = 0.0
    for i in range(len(timer_names)):
        t = 0.0
        if timer_names[i] in jin:
            t = jin[timer_names[i]]
            timer_names[i] = timer_names[i] + " (%6.2f)"%t
        timer_values.append(t)
        total_time += t
        
    print "total time for timer group ", itg, " ", total_time
    
    plot = matplotlib.pyplot.subplot("41%i"%(itg+1))
    box = plot.get_position()
    plot.set_position([box.x0, box.y0, box.width * 0.2, box.height])
    box = plot.get_position()
    
    ytics = [0]
    for i in range(len(timer_values)):
        ytics.append(ytics[i] + timer_values[i])
    
    plots = []
    for i in range(len(timer_values)):
        rgb = colorsys.hsv_to_rgb(i / float(len(timer_values)), 0.7, 0.9)
        c = "#%X%X%X"%(rgb[0]*255, rgb[1]*255, rgb[2]*255)
        plots.append(matplotlib.pyplot.bar(0, timer_values[i], 2, bottom=ytics[i], color=c))
    
    matplotlib.pyplot.xticks([], ())
    matplotlib.pyplot.yticks(ytics)
    matplotlib.pyplot.ylim([0, ytics[len(ytics)-1]])
    
    matplotlib.pyplot.legend(plots[::-1], timer_names[::-1], bbox_to_anchor=(1.2, 1), loc=2)

matplotlib.pyplot.savefig(fname+".pdf", format="pdf")

