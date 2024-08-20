# This script generates a basic header file from the sirius_api.cpp file
# A lot of assumptions are made: all functions are void, we only care about the extern "C",
# etc. In the end, the idea is to provide an input for Clang.jl wrapping

import re
import sys
 
#Only read the file from the extern "C" section beginning
is_extern = False
content = ""
with open("sirius_api.cpp", "r") as myfile:
    for line in myfile:
        if not is_extern:
            if line.startswith("extern"):
                is_extern = True

        if is_extern:
            content += line

#We want C-style complex types, not C++
content = content.replace("std::complex<double>", "double complex")

#A regex that matches all SIRIUS API calls: returns a void,
#starts with sirius_, and allows for line breaks
pattern = r'\bvoid\s+(?:\s*\n\s*)?sirius_\w+\s*\([^{]*\{'
matches = re.findall(pattern, content, re.DOTALL)

signatures = []
for match in matches:
    signatures.append(match.strip().replace("\n{", ";\n\n"))

#We also want to carry over the Fortran API info
pattern = r'@api begin(.*?)@api end'
matches = re.findall(pattern, content, re.DOTALL)
docs = []
for match in matches:
    docs.append("/*\n"+match.strip()+"\n*/\n")

with open("sirius_c_headers.h", "w") as myfile:
    myfile.write("#include <stdbool.h>\n")
    myfile.write("#include <complex.h>\n\n")
    for doc in docs:
        fname = doc.split("\n")[1][:-1]
        for signature in signatures:
            if fname+"(" in signature:
                myfile.write(doc)
                myfile.write(signature)
                break
