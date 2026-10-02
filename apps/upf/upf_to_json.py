import argparse
import json
import re
import os
from upf1_to_json import parse_upf1_from_string
from upf2_to_json import parse_upf2_from_string, add_paw_core_kinetic_density

def get_upf_version(upf):
    line = upf.split('\n')[0]
    if "<PP_INFO>" in line:
        return 1
    elif "UPF version" in line:
        return 2
    return 0

def parse_upf_from_string(upf_str):
    version = get_upf_version(upf_str)
    if version == 0:
        return None
    if version == 1:
        return parse_upf1_from_string(upf_str)
    if version == 2:
        return parse_upf2_from_string(upf_str)

def main():
    parser = argparse.ArgumentParser(description='Convert an UPF pseudopotential to SIRIUS JSON.')
    parser.add_argument('upf', help='Input UPF file')
    parser.add_argument('--paw-core-xml', help='Matching ATOMPAW XML with AE and pseudo core kinetic densities')
    parser.add_argument('-o', '--output', help='Output JSON file (default: ELEMENT.json)')
    args = parser.parse_args()
    fname = args.upf
    if not os.path.exists(fname):
        raise FileNotFoundError('invalid path for UPF file')

    with open(fname, 'r') as fh:
        upf_str = fh.read()

    pp_dict = parse_upf_from_string(upf_str)
    if pp_dict is None:
        parser.error('Unrecognized UPF version')
    if args.paw_core_xml:
        try:
            add_paw_core_kinetic_density(pp_dict, args.paw_core_xml)
        except (ValueError, RuntimeError, OSError) as error:
            parser.error(str(error))
    element = pp_dict['pseudo_potential']['header']['element']
    pp_dict['pseudo_potential']['header']['original_upf_file'] = fname

    with open(args.output or element + '.json', 'w') as fout:
        # Match comma, space, newline and an arbitrary number of spaces ',\s\n\s*' with the
        # following conditions: a digit before (?<=[0-9]) and a minus or a digit after (?=[-|0-9]).
        # Replace found sequence with comma and space.
        fout.write(re.sub(r"(?<=[0-9]),\s\n\s*(?=[-|0-9])", r", ", json.dumps(pp_dict, indent=2)))

if __name__ == "__main__":
    main()
