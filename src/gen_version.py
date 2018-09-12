import subprocess
import datetime

now = datetime.datetime.now()

with open("version.hpp", "w") as f:
    f.write("/** \\file version.hpp\n")
    f.write(" *  \\brief Auto-generated version file.\n")
    f.write(" *\n")
    f.write(" */\n")
    f.write("#ifndef __VERSION_HPP__\n")
    f.write("#define __VERSION_HPP__\n")
    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
    f.write("const char* const git_hash = \"%s\";\n"%(p.communicate()[0].strip()))
    p = subprocess.Popen(["git", "describe", "--all"], stdout=subprocess.PIPE)
    f.write("const char* const git_branchname = \"%s\";\n"%(p.communicate()[0].strip()))
    f.write("const char* const build_date = \"%s\";\n"%(now.strftime("%a, %e %b %Y %H:%M:%S")))
    f.write("#endif\n")

