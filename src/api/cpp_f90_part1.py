import sys
import re

# @fortran begin function                                    {type} {name} {doc-string}
# @fortran       argument {in|out|inout} {required|optional} {type} {name} {doc-string}
# [@fortran details]
# [detailed documentation]
# @fortran end

type_info = {
    'bool' : {
        'f_type' : 'logical',
        'c_type' : 'logical(C_BOOL)',
        'f2c'    : 'bool(%s)'
    },
    'void*' : {
        'f_type' : 'type(C_PTR)',
        'c_type' : 'type(C_PTR)'
    },
    'func' : {
        'f_type' : 'type(C_FUNPTR)',
        'c_type' : 'type(C_FUNPTR)'
    },
    'int' : {
        'f_type' : 'integer',
        'c_type' : 'integer(C_INT)'
    },
    'double' : {
        'f_type' : 'real(8)',
        'c_type' : 'real(C_DOUBLE)'
    },
    'string' : {
        'f_type' : 'character(*)',
        'c_type' : 'character(C_CHAR)',
        'f2c'    : 'string(%s)'
    },
    'complex' : {
        'f_type' : 'complex(8)',
        'c_type' : 'complex(C_DOUBLE)'
    }
}

def write_str_to_f90(o, string):
    p = 0
    while (True):
        p = string.find(',', p)
        # no more commas left in the string or string is short
        if p == -1 or len(string) <= 80:
            o.write("%s\n"%string)
            break;
        # position after comma
        p += 1
        if p >= 80:
            o.write("%s&\n&"%string[:p])
            string = string[p:]
            p = 0

def write_function_doc(o, func_doc, details, func_args):
    # write documentation header
    o.write("!> @brief %s\n"%func_doc)
    if details:
        o.write("!> @details\n")
        for l in details:
            o.write("!> %s\n"%l)

    for a in func_args:
        o.write("!> @param [%s] %s %s\n"%(a['intent'], a['name'], a['doc']))


def write_function(o, func_name, func_type, func_args, func_doc, details):
    # write documentation header
    write_function_doc(o, func_doc, details, func_args)

    # write declaration of function or subroutine
    string = 'subroutine ' if func_type == 'void' else 'function '
    string = string + func_name + '(' + ','.join([a['name'] for a in func_args]) + ')'
    if func_type != 'void':
        string = string + ' result(res)'
    write_str_to_f90(o, string)
    o.write('implicit none\n')

    # write list of arguments as seen by fortran.
    for a in func_args:
        o.write(type_info[a['type']]['f_type'])
        if not a['required']:
            o.write(', optional, target')
        #if a['type'] == 'func':
        #    o.write(', value')
        o.write(", intent(%s) :: %s"%(a['intent'], a['name']))
        o.write("\n")

    # result type if this is a function
    if func_type != 'void':
        o.write(type_info[func_type]['f_type'] + ' :: res\n')

    # declare auxiliary vaiables
    for a in func_args:
        t = a['type']
        if 'f2c' in type_info[t]:
            if t == "string":
                o.write("%s, target, allocatable :: %s_c_type(:)"%(type_info[t]['c_type'], a['name']))
            else:
                o.write("%s, target :: %s_c_type"%(type_info[t]['c_type'], a['name']))
            o.write("\n")

        if not a['required'] or t == 'string':
            o.write("type(C_PTR) :: %s_ptr\n"%a['name'])

    # create interface section
    o.write('interface\n')

    string = 'subroutine ' if func_type == 'void' else 'function '
    string = string + func_name + '_aux(' + ','.join([a['name'] for a in func_args]) + ')'
    if (func_type == 'void'):
        string = string + '&'
    else:
        string = string + ' result(res)&'
    write_str_to_f90(o, string)
    o.write("&bind(C, name=\"%s\")\n"%func_name)

    o.write('use, intrinsic :: ISO_C_BINDING\n')
    for a in func_args:
        t = a['type']
        # pass optional arguments explicitely by pointer
        if not a['required'] or t == 'string':
            o.write("type(C_PTR), value :: %s"%a['name'])
        else:
            o.write(type_info[a['type']]['c_type'])
            #!if a['type'] == 'string':
            #!    o.write(', dimension(*)')
            if a['type'] == 'func':
                o.write(', value')
            o.write(', intent(' + a['intent'] + ') :: ' + a['name'])
        o.write('\n')

    if func_type != 'void':
        o.write(type_info[func_type]['c_type'] + ' :: res\n')

    if func_type == 'void':
        o.write('end subroutine\n')
    else:
        o.write('end function\n')
    o.write('end interface\n\n')

    c_arg_names = []
    for a in func_args:
        t = a['type']
        n = a['name']

        if t == 'string':
            o.write("%s_ptr = C_NULL_PTR\n"%n)
            if not a['required']:
                o.write("if (present(%s)) then\n"%n)
            o.write("allocate(%s_c_type(len(%s)+1))\n"%(n, n))
            o.write(n + "_c_type = " +  type_info[t]['f2c']%n + "\n")
            o.write("%s_ptr = C_LOC(%s_c_type)\n"%(n, n))
            if not a['required']:
                o.write("endif\n")
            c_arg_names.append("%s_ptr"%n)
        else:
            if a['required']:
                if 'f2c' in type_info[t]:
                    s = n + '_c_type = ' + type_info[t]['f2c']%n + "\n"
                    o.write(s)
                    c_arg_names.append(n + '_c_type')
                else:
                    c_arg_names.append(n)
            else:
                o.write(n + '_ptr = C_NULL_PTR\n')
                c_arg_names.append(n + '_ptr')
                if 'f2c' in type_info[t]:
                    o.write("if (present(%s)) then\n"%n)
                    s = n + '_c_type = ' + type_info[t]['f2c']%n + "\n"
                    o.write('  ' + s)
                    o.write("  %s_ptr = C_LOC(%s_c_type)\n"%(n, n))
                    o.write("endif\n")
                else:
                    o.write('if (present('+a['name']+')) ' + a['name'] + '_ptr = C_LOC(' + a['name'] + ')\n\n')

    # make a function call
    string = 'call ' if func_type == 'void' else 'res = '
    string = string + func_name + '_aux(' + ','.join(c_arg_names) + ')'
    write_str_to_f90(o, string)

    for a in func_args:
        t = a['type']
        n = a['name']

        if t == 'string':
            o.write("if (allocated(%s_c_type)) deallocate(%s_c_type)\n"%(n, n))

    if func_type == 'void':
        o.write('end subroutine ')
    else:
        o.write('end function ')
    o.write(func_name + '\n\n')


def main():
    f = open(sys.argv[1], 'r') 
    
    #lines = '\n'.join(f.readlines())

    #for r in re.findall('(@fortran\s+begin\s+((\w|\W|\s)*?)@fortran\s+end)+', lines):
    #    print(r)
    #    print("-----------")

    o = open('generated.f90', 'w')
    o.write('! Warning! This file is automatically generated using cpp_f90.py script!\n\n')
    o.write('!> @file generated.f90\n')
    o.write('!! @brief Autogenerated interface to Fortran.\n')

    o.write('!\n')
    while (True):
        line = f.readline()
        if not line: break

        # parse @fortran begin function {type} {name} {doc-string}
        m = re.search('@fortran\s+begin\s+function\s+(\w+\*?)\s+(\w+)\s+((\w|\W|\s)*)', line)
        # match is successful
        if m:
            # we need to set the following variables:
            #   func_type
            #   func_name
            #   func_args
            #   func_doc
            #   details
            func_type = m.group(1)
            func_name = m.group(2)
            func_doc = m.group(3).strip()
            func_args = []
            details = []

            # parse strings until @fortran end is encountered
            while (True):
                line = f.readline()
                # check for @fortran details
                m = re.search('@fortran\s+details', line)
                if m:
                    while (True):
                        line = f.readline()
                        # check for @fortran end
                        m = re.search('@fortran', line)
                        if m: break
                        details.append(line.strip())

                # check for @fortran end
                m = re.search('@fortran\s+end', line)
                if m: break

                # parse @fortran argument {in|out|inout} {required|optional} {type} {name} {doc-string}
                m = re.search('@fortran\s+argument\s+(in|out|inout)\s+(required|optional)\s+(\w+\*?)\s+(\w+)\s+((\w|\W|\s)*)', line)
                if m: func_args.append({'type'     : m.group(3),
                                        'intent'   : m.group(1),
                                        'required' : m.group(2) == 'required',
                                        'name'     : m.group(4),
                                        'doc'      : m.group(5).strip()})

            write_function(o, func_name, func_type, func_args, func_doc, details)
            print('/*')
            print('@api begin')
            print("%s:"%func_name)
            if func_type != 'void':
                print("  return: %s"%func_type)
            print("  doc: %s"%func_doc)
            if details:
                print("  full_doc: %s"%details)
            print("  arguments:")
            for a in func_args:
                print("    %s:"%a['name'])
                print("      type: %s"%a['type'])
                print("      attr: %s, %s"%(a['intent'], 'required' if a['required'] else 'optional'))
                print("      doc: %s"%a['doc'])

            print('@api end')
            print('*/')
        else:
            print(line, end='')


    f.close()
    o.close()

if __name__ == "__main__":
    main()
