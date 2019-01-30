import sys
import os
import tarfile
import subprocess
import json

def aa():

    #cwdlibs = os.getcwd() + "/libs/"

    #if not os.path.exists(cwdlibs):
    #    os.makedirs(cwdlibs)

    new_env = os.environ.copy()
    #if 'FC' in new_env:
    #    new_env['F77'] = new_env['FC']

    print(new_env)

    #cmd = ["./configure"] + package["options"] + ["--prefix=%s"%prefix]
    #p = subprocess.Popen(cmd, cwd = './', env = new_env)
    #p.wait()

def main():
    aa()

if __name__ == "__main__":
    main()
