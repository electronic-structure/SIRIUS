import sys
import subprocess
import os
from subprocess import Popen, PIPE
import difflib
import argparse

#
# check formatting of the files with clang-format >= 22.0 (IndentPPDirectives: Leave)
#

parser = argparse.ArgumentParser(description='Check file formatting with clang-format')
parser.add_argument('-f', '--fmt', action='store_true', help='Apply formatting to the file(s)')
args = parser.parse_args()

env_copy = os.environ.copy()

all_files = [name.strip() for name in sys.stdin.readlines()]
files = []
for f in all_files:
    extension = os.path.splitext(f)[1][1:]
    if extension in ['h', 'hpp', 'cpp']:
        files.append(f)

print("files to check", files)

enable_shell = False

status = 0
for name in files:
    with open(name, 'r') as f:
        original = f.read()
    p = subprocess.Popen(["clang-format", "-style=file", name], shell=enable_shell, stdout=subprocess.PIPE)
    formatted = p.communicate()[0].decode('utf-8')
    # save formatted file
    if args.fmt:
        with open(name, 'w') as f:
            f.write(formatted)
    # report diff
    else:
        if original == formatted:
            print(f'{name}: OK')
        else:
            print(f'{name}: non-zero diff found')
            for line in difflib.unified_diff(original.splitlines(), formatted.splitlines(), fromfile='original', tofile='formatted'):
                print(line)
            status += 1

sys.exit(status)
