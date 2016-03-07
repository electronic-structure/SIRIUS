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

def parse_header(upf_dict, root):
    # header
    node = root.findall("./PP_HEADER")[0]
    upf_dict['header'] = {}
    upf_dict['header']['number_of_proj'] = int(node.attrib['number_of_proj'])
    upf_dict['header']['core_correction'] = bool(node.attrib['core_correction'])
    upf_dict['header']['element'] = node.attrib['element']
    upf_dict['header']['pseudo_type'] = node.attrib['pseudo_type']
    #upf_dict['header']['l_max'] = int(node.attrib['l_max'])
    upf_dict['header']['z_valence'] = float(node.attrib['z_valence'])

def parse_radial_grid(upf_dict, root):
    # radial grid
    node = root.findall("./PP_MESH/PP_R")[0]
    rg = [float(e) for e in str.split(node.text)]
    np = int(node.attrib['size'])
    if np != len(rg):
        print("Wrong number of radial points")
    upf_dict['radial_grid'] = rg
        
def parse_non_local(upf_dict, root):

    upf_dict['beta_projectors'] = []
        
    for i in range(upf_dict['header']['number_of_proj']):
        node = root.findall("./PP_NONLOCAL/PP_BETA.%i"%(i+1))[0]
        upf_dict['beta_projectors'].append({})
        beta = [float(e) for e in str.split(node.text)]
        upf_dict['beta_projectors'][i]['radial_function'] = beta
        upf_dict['beta_projectors'][i]['label'] = node.attrib['label']
        upf_dict['beta_projectors'][i]['angular_momentum'] = int(node.attrib['angular_momentum'])
        upf_dict['beta_projectors'][i]['cutoff_radius_index'] = int(node.attrib['cutoff_radius_index'])
        upf_dict['beta_projectors'][i]['cutoff_radius'] = float(node.attrib['cutoff_radius'])
        upf_dict['beta_projectors'][i]['ultrasoft_cutoff_radius'] = float(node.attrib['ultrasoft_cutoff_radius'])

    node = root.findall('./PP_NONLOCAL/PP_DIJ')[0]
    dij = [float(e) for e in str.split(node.text)]
    upf_dict['D_ion'] = [float(e) / 2 for e in str.split(node.text)] #convert to hartree

    if upf_dict['header']['pseudo_type'] != "USPP": return
    
    node = root.findall('./PP_NONLOCAL/PP_AUGMENTATION')[0]
    if node.attrib['q_with_l'] != 'T':
        print("Don't know how to parse this")
        sys.exit(0)

    upf_dict['augmentation'] = []

    nb = upf_dict['header']['number_of_proj']

    for i in range(nb):
        li = upf_dict['beta_projectors'][i]['angular_momentum']
        for j in range(i, nb):
            lj = upf_dict['beta_projectors'][j]['angular_momentum']
            for l in range(abs(li-lj), li+lj+1):
                if (li + lj + l) % 2 == 0:
                    node = root.findall("./PP_NONLOCAL/PP_AUGMENTATION/PP_QIJL.%i.%i.%i"%(i+1,j+1,l))[0]
                    qij = {}
                    qij['radial_function'] = [float(e) for e in str.split(node.text)]
                    qij['i'] = i
                    qij['j'] = j
                    qij['angular_momentum'] = int(node.attrib['angular_momentum'])
                    if l != qij['angular_momentum']:
                        print("Wrong angular momentum for Qij")
                        sys.exit(0)
                    upf_dict['augmentation'].append(qij)

def main():

    tree = ET.parse(sys.argv[1])
    root = tree.getroot()
    
    upf_dict = {}

    parse_header(upf_dict, root)
    parse_radial_grid(upf_dict, root)

    # non linear core correction
    if upf_dict['header']['core_correction']:
        node = root.findall("./PP_NLCC")[0]
        rc = [float(e) for e in str.split(node.text)]
        np = int(node.attrib['size'])
        if np != len(rc):
            print("Wrong number of points")
        upf_dict['core_charge_density'] = rc
    
    # local part of potential    
    node = root.findall("./PP_LOCAL")[0]
    vloc = [float(e) / 2 for e in str.split(node.text)] # convert to Ha
    np = int(node.attrib['size'])
    if np != len(vloc):
        print("Wrong number of points")
    upf_dict['local_potential'] = vloc

    # non-local part of potential
    parse_non_local(upf_dict, root)

    # rho
    node = root.findall("./PP_RHOATOM")[0]
    rho = [float(e) for e in str.split(node.text)]
    np = int(node.attrib['size'])
    if np != len(rho):
        print("Wrong number of points")
    upf_dict['total_charge_density'] = rho

    pp_dict = {}
    pp_dict["pseudo_potential"] = upf_dict

    fout = open(sys.argv[1] + ".json", "w")

    # Match comma, space, newline and an arbitrary number of spaces ',\s\n\s*' with the 
    # following conditions: a digit before (?<=[0-9]) and a minus or a digit after (?=[-|0-9]). 
    # Replace found sequence with comma and space.
    fout.write(re.sub(r"(?<=[0-9]),\s\n\s*(?=[-|0-9])", r", ", json.dumps(pp_dict, indent=2)))
    fout.close()

if __name__ == "__main__":
    main()
