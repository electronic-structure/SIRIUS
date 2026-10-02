# Copyright (c) 2016 Anton Kozhevnikov, Thomas Schulthess
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that
# the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
#    following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#    and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import math
import sys
import re
import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_header(upf_dict, root):
    # header
    node = root.findall("./PP_HEADER")[0]
    upf_dict['header'] = {}
    upf_dict['header']['number_of_proj'] = int(node.attrib['number_of_proj'])
    upf_dict['header']['core_correction'] = str2bool(
        node.attrib['core_correction'])
    upf_dict['header']['element'] = node.attrib['element'].strip()
    upf_dict['header']['pseudo_type'] = node.attrib['pseudo_type']
    upf_dict['header']['z_valence'] = float(node.attrib['z_valence'])
    upf_dict['header']['mesh_size'] = int(node.attrib['mesh_size'])
    upf_dict['header']['is_ultrasoft'] = str2bool(node.attrib['is_ultrasoft'])
    upf_dict['header']['number_of_wfc'] = int(node.attrib['number_of_wfc'])
    upf_dict['header']['spin_orbit'] = str2bool(node.attrib['has_so'])


def parse_radial_grid(upf_dict, root):
    # radial grid
    node = root.findall("./PP_MESH/PP_R")[0]
    rg = [float(e) for e in str.split(node.text)]
    try:
        np = int(node.attrib['size'])
        if np != len(rg):
            print("Wrong number of radial points")
    except KeyError:
        print('Warning: missing size field in attributes ' + str(node))
    upf_dict['radial_grid'] = rg


##########################################################################
#### Read non-local part: basis PS and AE (for PAW) functions,
#### beta(or p for PAW)-projectors, Qij augmentation coefs, Dij
##########################################################################
def parse_non_local(upf_dict, root):
    # ----------------------------------------------------
    # ------ Read beta (or p for PAW) - projectors  ------
    # ----------------------------------------------------
    upf_dict['beta_projectors'] = []

    proj_num = upf_dict['header']['number_of_proj']

    for i in range(proj_num):
        node = root.findall("./PP_NONLOCAL/PP_BETA.%i" % (i + 1))[0]
        beta = [float(e) for e in str.split(node.text)]

        try:
            # cutoff_radius_index is optional
            nr = int(node.attrib['cutoff_radius_index'])
        except KeyError:
            # ... and per the standard we should take the full list,
            # but we should cut off the long tail for numerical stability
            try:
                # find the first value from the back bigger than...
                nr = -next(idx for idx, val in enumerate(beta[::-1]) if val > 1e-80)
            except StopIteration:
                # if that fails, take the whole thing
                nr = len(beta)

        upf_dict['beta_projectors'].append({})
        upf_dict['beta_projectors'][i]['radial_function'] = beta[0:nr]
        if 'label' in node.attrib:
            upf_dict['beta_projectors'][i]['label'] = node.attrib['label']
        upf_dict['beta_projectors'][i]['angular_momentum'] = int(
            node.attrib['angular_momentum'])
        # upf_dict['beta_projectors'][i]['cutoff_radius_index'] = int(node.attrib['cutoff_radius_index'])
        # upf_dict['beta_projectors'][i]['cutoff_radius'] = float(node.attrib['cutoff_radius'])
        # if upf_dict['header']['is_ultrasoft']:
        #  upf_dict['beta_projectors'][i]['ultrasoft_cutoff_radius'] = float(node.attrib['ultrasoft_cutoff_radius'])
        if upf_dict['header']['spin_orbit']:
            node1 = root.findall("./PP_SPIN_ORB/PP_RELBETA.%i" % (i + 1))[0]
            upf_dict['beta_projectors'][i]['total_angular_momentum'] = float(
                node1.attrib['jjj'])

    # --------------------------
    # ------- Dij matrix -------
    # --------------------------
    node = root.findall('./PP_NONLOCAL/PP_DIJ')[0]
    # dij = [float(e) for e in str.split(node.text)]
    upf_dict['D_ion'] = [float(e) / 2
                         for e in str.split(node.text)]  # convert to hartree

    # if upf_dict['header']['pseudo_type'] == 'NC': return

    if not upf_dict['header']['is_ultrasoft']:
        return

    # ------------------------------------
    # ------- augmentation part: Qij  ----
    # ------------------------------------
    node = root.findall('./PP_NONLOCAL/PP_AUGMENTATION')[0]

    if not node.attrib['q_with_l'].lower() in ['t', 'true']:
        print("Don't know how to parse this 'q_with_l != T'")
        sys.exit(0)

    upf_dict['augmentation'] = []

    nb = upf_dict['header']['number_of_proj']

    # -----------------------------
    # --------- read Qij ----------
    # -----------------------------
    for i in range(nb):
        li = upf_dict['beta_projectors'][i]['angular_momentum']
        for j in range(i, nb):
            lj = upf_dict['beta_projectors'][j]['angular_momentum']
            for l in range(abs(li - lj), li + lj + 1):
                if (li + lj + l) % 2 == 0:
                    node = root.findall(
                        "./PP_NONLOCAL/PP_AUGMENTATION/PP_QIJL.%i.%i.%i" %
                        (i + 1, j + 1, l))[0]
                    qij = {}
                    qij['radial_function'] = [
                        float(e) for e in str.split(node.text)
                    ]
                    qij['i'] = i
                    qij['j'] = j
                    qij['angular_momentum'] = int(
                        node.attrib['angular_momentum'])
                    if l != qij['angular_momentum']:
                        print("Wrong angular momentum for Qij")
                        sys.exit(0)
                    upf_dict['augmentation'].append(qij)


####################################################
############# Read PAW data ########################
####################################################
def parse_PAW(upf_dict, root):

    if upf_dict['header']['pseudo_type'] != "PAW":
        return

    node = root.findall('./PP_NONLOCAL/PP_AUGMENTATION')[0]
    upf_dict['header']['cutoff_radius_index'] = int(
        node.attrib['cutoff_r_index'])

    upf_dict["paw_data"] = {}

    # -------------------------------------
    # ---- Read PP_Q and PP_MULTIPOLES ----
    # -------------------------------------
    node = root.findall('./PP_NONLOCAL/PP_AUGMENTATION/PP_Q')[0]
    upf_dict['paw_data']['aug_integrals'] = [
        float(e) for e in str.split(node.text)
    ]

    node = root.findall('./PP_NONLOCAL/PP_AUGMENTATION/PP_MULTIPOLES')[0]
    upf_dict['paw_data']['aug_multipoles'] = [
        float(e) for e in str.split(node.text)
    ]

    # ----------------------------------------
    # ---- Read AE and PS basis wave functions
    # ----------------------------------------
    nb = upf_dict['header']['number_of_proj']

    # ----- Read AE wfc -----
    upf_dict['paw_data']['ae_wfc'] = []

    for i in range(nb):
        wfc = {}
        node = root.findall("./PP_FULL_WFC/PP_AEWFC.%i" % (i + 1))[0]
        wfc['radial_function'] = [float(e) for e in str.split(node.text)]
        wfc['angular_momentum'] = int(node.attrib['l'])
        # wfc['label'] = node.attrib['label']
        # wfc['index'] =  int(node.attrib['index']) - 1
        upf_dict['paw_data']['ae_wfc'].append(wfc)

    # ----- Read PS wfc -----
    upf_dict['paw_data']['ps_wfc'] = []

    for i in range(nb):
        wfc = {}
        node = root.findall("./PP_FULL_WFC/PP_PSWFC.%i" % (i + 1))[0]
        wfc['radial_function'] = [float(e) for e in str.split(node.text)]
        wfc['angular_momentum'] = int(node.attrib['l'])
        # wfc['label'] = node.attrib['label']
        # wfc['index'] =  int(node.attrib['index']) - 1
        upf_dict['paw_data']['ps_wfc'].append(wfc)

    # ------ Read PP_PAW section: occupation, AE_NLCC, AE_VLOC
    node = root.findall("./PP_PAW")[0]
    try:
        upf_dict['header']["paw_core_energy"] = float(
            node.attrib['core_energy']) / 2  # convert to Ha
    except KeyError:
        print('WARNING: PP_PAW has no core_energy set!')

    node = root.findall("./PP_PAW/PP_OCCUPATIONS")[0]
    if 'size' in node.attrib:
        if upf_dict['header']['number_of_proj'] != int(node.attrib['size']):
            print('WARNING: number_of_proj != size(PP_OCCUPATIONS)')
    size = upf_dict['header']['number_of_proj']

    # ---- occupation
    node = root.findall("./PP_PAW/PP_OCCUPATIONS")[0]
    upf_dict['paw_data']['occupations'] = [
        float(e) for e in str.split(node.text)
    ]
    try:
        size = int(node.attrib['size'])
    except KeyError:
        size = len(upf_dict['paw_data']['occupations'])
        print('WARNING: the size attribute in PP_PAW/PP_OCCUPATIONS is missing')
    assert len(upf_dict['paw_data']['occupations']) == size

    # ---- Read AE core correction (density of core charge)
    if 'size' in node.attrib:
        if upf_dict['header']['mesh_size'] != int(node.attrib['size']):
            print('WARNING: mesh_size != size(PP_AE_NLCC)')
    node = root.findall("./PP_PAW/PP_AE_NLCC")[0]
    size = upf_dict['header']['mesh_size']

    upf_dict['paw_data']['ae_core_charge_density'] = [
        float(e) for e in str.split(node.text)
    ]
    try:
        size = int(node.attrib['size'])
    except KeyError:
        size = len(upf_dict['paw_data']['ae_core_charge_density'])
        print('WARNING: the size attribute in PP_PAW/PP_AE_NLCC is missing')
    assert len(upf_dict['paw_data']['ae_core_charge_density']) == size

    # ---- Read AE local potential
    node = root.findall("./PP_PAW/PP_AE_VLOC")[0]
    if 'size' in node.attrib:
        if upf_dict['header']['mesh_size'] != int(node.attrib['size']):
            print('WARNING: mesh_size != size(PP_AE_VLOC)')
    size = upf_dict['header']['mesh_size']

    upf_dict['paw_data']['ae_local_potential'] = [
        float(e) / 2 for e in str.split(node.text)
    ]  # convert to Ha
    try:
        size = int(node.attrib['size'])
    except KeyError:
        size = len(upf_dict['paw_data']['ae_local_potential'])
        print('WARNING: the size attribute in PP_PAW/PP_AE_VLOC is missing')
    assert len(upf_dict['paw_data']['ae_local_potential']) == size
####################################################
############# Read starting wave functions #########
####################################################
def parse_pswfc(upf_dict, root):
    # if upf_dict['header']['pseudo_type'] != 'NC': return

    upf_dict['atomic_wave_functions'] = []

    for i in range(upf_dict['header']['number_of_wfc']):
        wfc = {}
        node = root.findall("./PP_PSWFC/PP_CHI.%i" % (i + 1))[0]
        wfc['radial_function'] = [float(e) for e in str.split(node.text)]
        wfc['angular_momentum'] = int(node.attrib['l'])
        # wfc['label'] = node.attrib['label']
        wfc['occupation'] = float(node.attrib['occupation'])
        if upf_dict['header']['spin_orbit']:
            node = root.findall("./PP_SPIN_ORB/PP_RELWFC.%i" % (i + 1))[0]
            wfc['total_angular_momentum'] = float(node.attrib['jchi'])
        upf_dict['atomic_wave_functions'].append(wfc)


####################################################
############# Spin orbit coupling #################
####################################################
def parse_SpinOrbit(upf_dict, root):
    if not upf_dict['header']['spin_orbit']: return

    # Spin orbit informations for the projectors

    proj_num = upf_dict['header']['number_of_proj']
    for i in range(proj_num):
        node = root.findall("./PP_SPIN_ORB/PP_RELBETA.%i" % (i + 1))[0]
        upf_dict['beta_projectors'][i]['angular_momentum'] = float(
            node.attrib['lll'])
        upf_dict['beta_projectors'][i]['total_angular_momentum'] = float(
            node.attrib['jjj'])

    # spin orbit information for the AEWFC


#    wfc_num = upf_dict['header']['number_of_wfc']
#    for i in range(wfc_num):
#      node = root.findall("./PP_SPIN_ORB/PP_RELWFC.%i"%(i+1))[0]
#      upf_dict['paw_data']['ae_wfc']['ae_wfc_rel'] = float(node)
#      upf_dict['paw_data']['ae_wfc']['total_angular_momentum'] = float(node('jchi'))
#      upf_dict['paw_data']['ps_wfc']['total_angular_momentum'] = float(node('jchi'))


def parse_upf2_from_string(upf2_str):

    # fix string
    upf2_str = upf2_str.replace("&", "")

    root = ET.fromstring(upf2_str)

    upf_dict = {}

    parse_header(upf_dict, root)
    parse_radial_grid(upf_dict, root)

    # non linear core correction
    if upf_dict['header']['core_correction']:
        node = root.findall("./PP_NLCC")[0]
        rc = [float(e) for e in str.split(node.text)]
        try:
            np = int(node.attrib['size'])
            if np != len(rc):
                print("Wrong number of points")
        except KeyError:
            print('Warning: missing size field in attributes ' + str(node))
        upf_dict['core_charge_density'] = rc

    # local part of potential
    node = root.findall("./PP_LOCAL")[0]
    vloc = [float(e) / 2 for e in str.split(node.text)]  # convert to Ha
    try:
        np = int(node.attrib['size'])
        if np != len(vloc):
            print("Wrong number of points")
    except KeyError:
        print('Warning: missing size field in attributes ' + str(node))
    upf_dict['local_potential'] = vloc

    # non-local part of potential
    parse_non_local(upf_dict, root)

    # parse PAW data
    parse_PAW(upf_dict, root)

    # parse pseudo wavefunctions
    parse_pswfc(upf_dict, root)

    # parse data for spin orbit coupling
    parse_SpinOrbit(upf_dict, root)

    # rho
    node = root.findall("./PP_RHOATOM")[0]
    rho = [float(e) for e in str.split(node.text)]
    try:
        np = int(node.attrib['size'])
        if np != len(rho):
            print("Wrong number of points")
    except KeyError:
        print('Warning: missing size field in attributes ' + str(node))
    upf_dict['total_charge_density'] = rho

    pp_dict = {}
    pp_dict["pseudo_potential"] = upf_dict

    return pp_dict


def parse_upf2_from_file(upf2_fname):
    with open(upf2_fname) as inpf:
        upf2_str = inpf.read()
    return parse_upf2_from_string(upf2_str)


def _nonnegative_values(values, name):
    values = [float(v) for v in values]
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(name + " must contain finite, nonnegative values")
    return values


def _radial_grid(values, name):
    values = _nonnegative_values(values, name)
    if len(values) < 3 or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError(name + " must have at least three strictly increasing points")
    return values


def _resample_core(grid, values, target):
    if target[0] < grid[0] or target[-1] > grid[-1]:
        raise ValueError("PAW-XML radial grid does not cover the UPF grid")
    if target == grid:
        return values[:]

    # GSL is already a SIRIUS dependency. Steffen interpolation preserves
    # positivity without introducing extra extrema in the core profiles.
    import ctypes as ct
    from ctypes.util import find_library

    library = find_library("gsl")
    if library is None:
        raise RuntimeError("PAW-XML resampling requires the GSL shared library on the library search path")
    gsl = ct.CDLL(library)
    ptr = ct.POINTER(ct.c_double)
    gsl.gsl_interp_alloc.argtypes = [ct.c_void_p, ct.c_size_t]
    gsl.gsl_interp_alloc.restype = ct.c_void_p
    gsl.gsl_interp_init.argtypes = [ct.c_void_p, ptr, ptr, ct.c_size_t]
    gsl.gsl_interp_eval_e.argtypes = [ct.c_void_p, ptr, ptr, ct.c_double, ct.c_void_p, ptr]
    gsl.gsl_interp_free.argtypes = [ct.c_void_p]
    gsl.gsl_interp_free.restype = None
    x = (ct.c_double * len(grid))(*grid)
    y = (ct.c_double * len(values))(*values)
    interp = gsl.gsl_interp_alloc(ct.c_void_p.in_dll(gsl, "gsl_interp_steffen"), len(grid))
    if not interp:
        raise MemoryError("Cannot allocate PAW-XML radial interpolator")
    try:
        if gsl.gsl_interp_init(interp, x, y, len(grid)) != 0:
            raise ValueError("Cannot initialize PAW-XML radial interpolator")
        result = []
        for radius in target:
            value = ct.c_double()
            if gsl.gsl_interp_eval_e(interp, x, y, radius, None, ct.byref(value)) != 0:
                raise ValueError("Cannot interpolate PAW-XML core field")
            result.append(value.value)
        return _nonnegative_values(result, "Interpolated core field")
    finally:
        gsl.gsl_interp_free(interp)


def add_paw_core_kinetic_density(pp_dict, xml_file):
    """Import positive core tau from the matching ATOMPAW PAW-XML dataset.

    Both core charge profiles must match the UPF to 1e-6 of their peak after
    interpolation. This checks compatibility, not provenance: the caller must
    supply the XML and UPF exported from the same atomic calculation.
    """
    pp = pp_dict['pseudo_potential']
    if pp['header']['pseudo_type'] != 'PAW':
        raise ValueError("PAW core XML requires a PAW UPF dataset")
    try:
        root = ET.parse(xml_file).getroot()
    except ET.ParseError as error:
        raise ValueError("Invalid PAW-XML: " + str(error)) from error

    def node(path):
        nodes = root.findall(path)
        if len(nodes) != 1:
            raise ValueError("PAW-XML must contain exactly one " + path)
        return nodes[0]

    if root.tag != 'paw_dataset' or not node('generator').get('name', '').lower().startswith('atompaw-'):
        raise ValueError("Expected an ATOMPAW paw_dataset XML file")
    atom = node('atom')
    z, core, valence = _nonnegative_values(
        [atom.get('Z', 'nan'), atom.get('core', 'nan'), atom.get('valence', 'nan')], 'Atomic charges')
    upf_valence = _nonnegative_values([pp['header']['z_valence']], 'UPF valence charge')[0]
    if (z == 0 or atom.get('symbol') != pp['header']['element'] or
            abs(valence - upf_valence) > 1e-8 or
            abs(core + valence - z) > 1e-8):
        raise ValueError("PAW-XML element or electron counts do not match the UPF")
    target = _radial_grid(pp['radial_grid'], 'UPF radial grid')
    grids = {}
    for radial in root.findall('radial_grid'):
        name = radial.get('id')
        if not name or name in grids:
            raise ValueError("PAW-XML radial grids must have unique IDs")
        grids[name] = _radial_grid((radial.findtext('values') or '').split(), 'PAW-XML radial grid')

    def field(tag):
        data = node(tag)
        grid = grids.get(data.get('grid'))
        if grid is None:
            raise ValueError("Unknown radial grid for " + tag)
        values = _nonnegative_values((data.text or '').split(), tag)
        if len(values) != len(grid):
            raise ValueError("Wrong number of radial values for " + tag)
        # ATOMPAW stores Y_00 coefficients. SIRIUS uses full spherical fields
        # in atomic units, including tau = sum_i f_i |grad psi_i|^2 / 2.
        values = [v / math.sqrt(4 * math.pi) for v in values]
        return _resample_core(grid, values, target)

    def matching(values, original, name):
        original = _nonnegative_values(original, name)
        if len(original) != len(target):
            raise ValueError("Wrong number of UPF radial values for " + name)
        if max(abs(a - b) for a, b in zip(values, original)) > 1e-6 * max(original):
            raise ValueError("PAW-XML and UPF disagree on " + name)

    paw = pp['paw_data']
    ps_core = pp.get('core_charge_density')
    if ps_core is None and not pp['header']['core_correction']:
        ps_core = [0.0] * len(target)
    if ps_core is None:
        raise ValueError("UPF core correction is enabled but its core charge density is missing")
    for tag, original in [('ae_core_density', paw['ae_core_charge_density']),
                          ('pseudo_core_density', ps_core)]:
        matching(field(tag), original, tag)
    ae_tau = field('ae_core_kinetic_energy_density')
    ps_tau = field('pseudo_core_kinetic_energy_density')
    if (core > 0) != any(paw['ae_core_charge_density']):
        raise ValueError("PAW-XML core count disagrees with the UPF AE core charge density")
    if (core > 0 and not any(ae_tau)) or (any(ps_core) and not any(ps_tau)):
        raise ValueError("Nonzero core charge requires nonzero supplied core kinetic density")
    if core == 0 and (any(ae_tau) or any(ps_tau)):
        raise ValueError("A core-free dataset must have zero core kinetic density")
    for owner, key, values in [(paw, 'ae_core_kinetic_density', ae_tau),
                               (pp, 'core_kinetic_density', ps_tau)]:
        if key in owner:
            matching(values, owner[key], key)
    # Do not modify the dataset until every field and compatibility check passes.
    paw['ae_core_kinetic_density'] = ae_tau
    pp['core_kinetic_density'] = ps_tau


def main():

    pp_dict = parse_upf2_from_file(sys.argv[1])

    fout = open(sys.argv[1] + ".json", "w")

    # Match comma, space, newline and an arbitrary number of spaces ',\s\n\s*' with the
    # following conditions: a digit before (?<=[0-9]) and a minus or a digit after (?=[-|0-9]).
    # Replace found sequence with comma and space.
    fout.write(
        re.sub(r"(?<=[0-9]),\s\n\s*(?=[-|0-9])", r", ",
               json.dumps(pp_dict, indent=2)))
    fout.close()


if __name__ == "__main__":
    main()
