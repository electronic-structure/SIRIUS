import sys
import subprocess
import os
from subprocess import Popen, PIPE
import difflib

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
    p2 = subprocess.Popen(["clang-format", "-style=file"],            shell=enable_shell, stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["sed", r"s/\/\/ *#pragma omp/#pragma omp/g"],     shell=enable_shell, stdin=p2.stdout, stdout=subprocess.PIPE)
    formatted = p3.communicate()[0].decode('utf-8')
    p3.wait()
    if (original == formatted):
        print("%s: OK"%name)
    else:
        print("%s: non-zero diff found"%name)
        for line in difflib.unified_diff(original.splitlines(), formatted.splitlines(), fromfile='original', tofile='formatted'):
            print(line)
        status += 1

sys.exit(status)
