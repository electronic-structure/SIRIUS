import sys
import subprocess
import datetime
import json

now = datetime.datetime.now()

request_str = 'https://api.github.com/repos/electronic-structure/SIRIUS/tags'

def to_string(s):
    if isinstance(s, str):
        return s
    else:
        return s.decode('utf-8')

with open("version.hpp", "w") as f:
    f.write("/** \\file version.hpp\n")
    f.write(" *  \\brief Auto-generated version file.\n")
    f.write(" */\n")
    f.write("#ifndef __VERSION_HPP__\n")
    f.write("#define __VERSION_HPP__\n")

    vstr = ''

    sha_str = ''
    try:
        p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
        sha_str = p.communicate()[0].strip()
    except:
        # get version string from the file
        with open('VERSION') as vf:
            vstr = vf.readline().strip()
        # python2 and python3 handle URL requests differently
        if sys.version_info < (3, 0):
            import urllib2
            data = json.load(urllib2.urlopen(request_str))
        else:
            import urllib.request
            req = urllib.request.Request(request_str)
            data = json.load(urllib.request.urlopen(req))
        # serach for a version tag
        for e in data:
            if e['name'] == 'v' + vstr:
                sha_str = e['commit']['sha']
                break

    f.write("const char* const git_hash = \"%s\";\n"%to_string(sha_str))

    branch_name = ''
    try:
        p = subprocess.Popen(["git", "describe", "--all"], stdout=subprocess.PIPE)
        branch_name = p.communicate()[0].strip()
    except:
        branch_name = 'release tag v%s'%vstr
    f.write("const char* const git_branchname = \"%s\";\n"%to_string(branch_name))

    f.write("const char* const build_date = \"%s\";\n"%(now.strftime("%a, %e %b %Y %H:%M:%S")))
    f.write("#endif\n")

