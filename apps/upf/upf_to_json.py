import json
import sys
import re
import upf1_to_json
import upf2_to_json

def get_upf_version(file_name):
    with open(file_name) as inp:
        line = inp.readline()
    if "<PP_INFO>" in line:
        return 1
    elif "UPF version" in line:
        return 2
    return 0

def parse_upf_from_file(file_name):
    version = get_upf_version(file_name)
    if version == 0:
        return None
    if version == 1:
        return upf1_to_json.parse_upf1_from_file(file_name)
    if version == 2:
        return upf2_to_json.parse_upf2_from_file(file_name)

def main():
    pp_dict = parse_upf_from_file(sys.argv[1])

    fout = open(sys.argv[1] + ".json", "w")

    # Match comma, space, newline and an arbitrary number of spaces ',\s\n\s*' with the
    # following conditions: a digit before (?<=[0-9]) and a minus or a digit after (?=[-|0-9]).
    # Replace found sequence with comma and space.
    fout.write(re.sub(r"(?<=[0-9]),\s\n\s*(?=[-|0-9])", r", ", json.dumps(pp_dict, indent=2)))
    fout.close()

if __name__ == "__main__":
    main()
