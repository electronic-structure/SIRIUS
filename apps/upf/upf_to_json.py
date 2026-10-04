import json
import sys
import re
import os
from upf1_to_json import parse_upf1_from_string
from upf2_to_json import parse_upf2_from_string

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

    fname = sys.argv[1]
    if not os.path.exists(fname):
        raise FileNotFoundError('invalid path for UPF file')

    with open(sys.argv[1], 'r') as fh:
        upf_str = fh.read()

    pp_dict = parse_upf_from_string(upf_str)
    element = pp_dict['pseudo_potential']['header']['element']
    pp_dict['pseudo_potential']['header']['original_upf_file'] = sys.argv[1]

    with open(element + '.json', 'w') as fout:
        # Match comma, space, newline and an arbitrary number of spaces ',\s\n\s*' with the
        # following conditions: a digit before (?<=[0-9]) and a minus or a digit after (?=[-|0-9]).
        # Replace found sequence with comma and space.
        fout.write(re.sub(r"(?<=[0-9]),\s\n\s*(?=[-|0-9])", r", ", json.dumps(pp_dict, indent=2)))

if __name__ == "__main__":
    main()
