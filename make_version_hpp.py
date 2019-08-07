import sys
import subprocess
import datetime
import json
import os
import datetime

now = datetime.datetime.now()

# The SIRIUS releases are tagged with `vX.Y.Z` schema, where X is a major version,
# Y is a minor version and Z is a revision.

# Use github REST API to query the tags
request_str = 'https://api.github.com/repos/electronic-structure/SIRIUS/tags'

# Python2 compatibility
try:
    FileNotFoundError
except:
    FileNotFoundError = IOError

def to_string(s):
    """
    Convert to ASCII sring.
    """
    if isinstance(s, str):
        return s
    else:
        return s.decode('utf-8')


def get_sha(vstr, dirname):
    """
    Get SHA hash of the code.
    Try to call git. If git command is not found or this is not a git repository, try to read
    the file VERSION to get the tag name. If none found, return an empty string.
    """

    sha_str = ""

    try:
        p = subprocess.Popen(["git", "rev-parse", "HEAD"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             cwd=dirname)
        sha_str = p.communicate()[0].strip()
        if p.returncode is not 0:
            raise RuntimeError
        return to_string(sha_str)
    except (RuntimeError, FileNotFoundError, OSError):
        # python2 and python3 handle URL requests differently
        try:
            if sys.version_info < (3, 0):
                from urllib2 import urlopen, URLError
                data = json.load(urlopen(request_str))
            else:
                from urllib.request import Request, urlopen, URLError
                req = Request(request_str)
                data = json.loads(urlopen(req).read().decode('utf-8'))
            # search for a version tag
            for e in data:
                if e['name'] == 'v' + vstr:
                    sha_str = e['commit']['sha']
                    break
            return to_string(sha_str)
        except URLError:
            return 'GIT_SHA_UNKNOWN'
    except:
        return 'GIT_SHA_UNKNOWN'


def get_branch(sha_str, vstr):
    """
    Get name of the branch. If git command failed but SHA is found, this is a release version
    """
    branch_name = ""
    try:
        p = subprocess.Popen(["git", "describe", "--all"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        branch_name = p.communicate()[0].strip()
        if p.returncode is not 0:
            raise RuntimeError
    except (RuntimeError, FileNotFoundError):
        if sha_str:
            branch_name = 'release tag v%s' % vstr
    return to_string(branch_name)


def main():
    print("/** \\file version.hpp")
    print(" *  \\brief Auto-generated version file.")
    print(" */\n")
    print("#ifndef __VERSION_HPP__")
    print("#define __VERSION_HPP__")

    fname = sys.argv[1]  # path to VERSION file

    version_str = ""

    major_version = -1
    minor_version = -1
    revision = -1

    # get version string from the file
    try:
        with open(sys.argv[1]) as vf:
            version_str = vf.readline().strip()
            version = version_str.split('.')
            major_version = int(version[0])
            minor_version = int(version[1])
            revision = int(version[2])
    except OSError:
        pass
    sha_str = get_sha(version_str, os.path.dirname(fname))
    branch_name = get_branch(sha_str, version_str)

    print("const char* const git_hash = \"%s\";" % sha_str)
    print("const char* const git_branchname = \"%s\";" % branch_name)
    #print("const char* const build_date = \"%s\";"%(now.strftime("%a, %e %b %Y %H:%M:%S")))
    print("const int major_version = %i;" % major_version)
    print("const int minor_version = %i;" % minor_version);
    print("const int revision      = %i;" % revision);
    print("#endif")

if __name__ == "__main__":
    main()
