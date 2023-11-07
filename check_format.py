import sys
import subprocess
import os
from subprocess import Popen, PIPE
import difflib
import argparse

parser = argparse.ArgumentParser(description='Check file formatting with clang-format')
parser.add_argument('-f', '--fmt', action='store_true', help='Apply formatting to the file(s)')
#parser.add_argument('-o', '--output', help='Output folder')
#parser.add_argument('-a', '--action', help='Do one of the following actions: dry run, copy files, move files', default='dry', choices=["dry", "copy", "move"])
#parser.add_argument('-d', '--date', action='store_true', help='Move/copy files to final destination folder based on their date stamps')
#parser.add_argument('-e', '--ext', help='Allowed file extensions', nargs="+", type=str, default=['jpg', 'jpeg', 'dng'])
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
    p1 = subprocess.Popen(["sed", r"s/#pragma omp/\/\/#pragma omp/g", name], shell=enable_shell, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["clang-format", "-style=file"],                   shell=enable_shell, stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["sed", r"s/\/\/ *#pragma omp/#pragma omp/g"],     shell=enable_shell, stdin=p2.stdout, stdout=subprocess.PIPE)
    formatted = p3.communicate()[0].decode('utf-8')
    p3.wait()
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
