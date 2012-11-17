import matplotlib.pyplot
import scipy
import colorsys
import json

matplotlib.pyplot.figure(1, figsize=(10,14))

fin = open("timers.json", "r")
jin = json.load(fin)
fin.close()

timer_names = [
    "sirius::Global::generate_radial_functions",
    "sirius::Global::generate_radial_integrals", 
    "sirius::Density::find_eigen_states",
    "sirius::Density::generate",
    "sirius::Potential::generate_effective_potential",
    "elk::symmetrization",
    "elk::mixer"
]

timer_values = []
for i in range(len(timer_names)):
    timer_values.append(jin[timer_names[i]])

timer_values[2] = timer_values[2] - timer_values[1] - timer_values[0]

plot1 = matplotlib.pyplot.subplot(411)
box = plot1.get_position()
plot1.set_position([box.x0, box.y0, box.width * 0.2, box.height])
box = plot1.get_position()

ytics = [0]
for i in range(len(timer_values)):
    #ytics.append(int((ytics[i] + timer_values[i]) * 100) / 100.0)
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


timer_names = [
    "sirius::Potential::poisson",
    "sirius::Potential::xc"
]

timer_values = []
for i in range(len(timer_names)):
    timer_values.append(jin[timer_names[i]])

plot4 = matplotlib.pyplot.subplot(412)
box = plot4.get_position()
plot4.set_position([box.x0, box.y0, box.width * 0.2, box.height])
box = plot4.get_position()

ytics = [0]
for i in range(len(timer_values)):
    #ytics.append(int((ytics[i] + timer_values[i]) * 100) / 100.0)
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

timer_names = [
    "sirius::kpoint::generate_matching_coefficients",
    "sirius::Band::set_h",
    "sirius::Band::set_o",
    "sirius::Band::solve_fv:hegv<impl>",
    "sirius::kpoint::generate_scalar_wave_functions",
    "sirius::kpoint::generate_spinor_wave_functions"
]

timer_values = []
for i in range(len(timer_names)):
    timer_values.append(jin[timer_names[i]])

#matplotlib.pyplot.title("Iteration")

plot2 = matplotlib.pyplot.subplot(413)
box = plot2.get_position()
plot2.set_position([box.x0, box.y0, box.width * 0.2, box.height])
box = plot2.get_position()

ytics = [0]
for i in range(len(timer_values)):
    #ytics.append(int((ytics[i] + timer_values[i]) * 100) / 100.0)
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


timer_names = [
    "sirius::reciprocal_lattice::init",
    "sirius::sirius_step_func::init",
    "sirius::unit_cell::get_symmetry",
    "sirius::geometry::find_nearest_neighbours",
    "sirius::kpoint::initialize",
    "sirius::Potential::Potential",
    "sirius::AtomType::solve_free_atom"
]

timer_values = []
for i in range(len(timer_names)):
    timer_values.append(jin[timer_names[i]])

#matplotlib.pyplot.title("Iteration")

plot3 = matplotlib.pyplot.subplot(414)
box = plot3.get_position()
plot3.set_position([box.x0, box.y0, box.width * 0.2, box.height])
box = plot3.get_position()

ytics = [0]
for i in range(len(timer_values)):
    #ytics.append(int((ytics[i] + timer_values[i]) * 100) / 100.0)
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







matplotlib.pyplot.savefig("timers.pdf", format="pdf")
