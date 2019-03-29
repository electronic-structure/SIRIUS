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
        print('Warning missing size field in attributes')
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
        nr = int(node.attrib['cutoff_radius_index'])
        upf_dict['beta_projectors'].append({})
        beta = [float(e) for e in str.split(node.text)]
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

    if node.attrib['q_with_l'] != 'T':
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
    size = int(node.attrib['size'])

    # ---- occupation
    for i in range(size):
        upf_dict['paw_data']['occupations'] = [
            float(e) for e in str.split(node.text)
        ]

    # ---- Read AE core correction (density of core charge)
    node = root.findall("./PP_PAW/PP_AE_NLCC")[0]
    size = int(node.attrib['size'])

    for i in range(size):
        upf_dict['paw_data']['ae_core_charge_density'] = [
            float(e) for e in str.split(node.text)
        ]

    # ---- Read AE local potential
    node = root.findall("./PP_PAW/PP_AE_VLOC")[0]
    size = int(node.attrib['size'])

    for i in range(size):
        upf_dict['paw_data']['ae_local_potential'] = [
            float(e) / 2 for e in str.split(node.text)
        ]  # convert to Ha


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
        print('Warning missing size field in attributes ' + str(node))
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
