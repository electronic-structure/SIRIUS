import sys
import re
import yaml

type_info = {
    'bool' : {
        'f_type' : 'logical',
        'c_type' : 'logical(C_BOOL)'
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
        'c_type' : 'character(C_CHAR)'
    },
    'complex' : {
        'f_type' : 'complex(8)',
        'c_type' : 'complex(C_DOUBLE)'
    }
}

# named constants
i_arg_name = 0
i_arg_type = 1
i_arg_intent = 2
i_arg_requirement = 3
i_arg_doc = 4
i_arg_f_type = 5
i_arg_c_type = 6
i_arg_dims = 7

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


def get_arg_attr(arg):
    if 'attr' in arg:
        attr = arg['attr'].replace(' ', '').split(',', 2)
        if len(attr) > 3:
            raise Exception("too many attributes")
        if attr[0] not in ('in', 'out', 'inout'):
            raise Exception('wrong intent attribute')
        if attr[1] not in ('optional', 'required'):
            raise Exception('wrong requirement attribute')
        if len(attr) == 2:
            attr.append("scalar")
        return attr
    else:
        raise Exception('argment attributes are not provided')


def write_function_doc(o, func_doc, details, func_args):
    '''
    Write a comment block with function documentation
    '''
    # write documentation header
    o.write("!> @brief %s\n"%func_doc)
    if details:
        o.write("!> @details\n")
        for l in details.split('\n'):
            o.write(f"!> {l}\n")
        #for l in details:
        #    o.write("!> %s\n"%l)

    for a in func_args:
        o.write("!> @param [%s] %s %s\n"%(a[i_arg_intent], a[i_arg_name], a[i_arg_doc]))

def write_interface(o, func_name, func_type, args):
    '''
    Write interface block
    '''

    # create interface section
    o.write('interface\n')

    string = 'subroutine ' if func_type == 'void' else 'function '
    string = string + func_name + '_aux(' + ','.join([a[i_arg_name] for a in args]) + ')'

    if (func_type == 'void'):
        string = string + '&'
    else:
        string = string + ' result(res)&'
    write_str_to_f90(o, string)
    o.write("&bind(C, name=\"%s\")\n"%func_name)

    o.write('use, intrinsic :: ISO_C_BINDING\n')
    # declare arguments of the interface (as seen by C code)
    for a in args:
        if a[i_arg_type] == 'func':
            o.write('type(C_FUNPTR)')
        else:
            o.write('type(C_PTR)')
        o.write(", value :: %s\n"%a[i_arg_name])

    if func_type != 'void':
        o.write("%s :: res\n"%type_info[func_type]['c_type'])

    if func_type == 'void':
        o.write('end subroutine\n')
    else:
        o.write('end function\n')

    o.write('end interface\n!\n')


def write_function(o, func_name, func_type, func_args, func_doc, details):
    '''
    Write the intefrace to the function
    '''

    # build internal list of function arguments
    args = []
    if func_args:
        for a in func_args:
            attr = get_arg_attr(func_args[a])
            is_optional = (attr[1] == 'optional')
            t = func_args[a]['type']
            # check that function pointers are always in and required
            if t == 'func' and not (attr[0] == 'in' and attr[1] == 'required'):
                raise Exception("wroing attributes for function pointer")
            doc = func_args[a]['doc']
            args.append((a, t, attr[0], attr[1], doc, type_info[t]['f_type'], type_info[t]['c_type'], attr[2]))

    # write documentation header
    write_function_doc(o, func_doc, details, args)

    # write declaration of function or subroutine
    string = 'subroutine ' if func_type == 'void' else 'function '
    string = string + func_name + '(' + ','.join([a[i_arg_name] for a in args]) + ')'
    if func_type != 'void':
        string = string + ' result(res)'
    write_str_to_f90(o, string)
    o.write('implicit none\n')
    o.write('!\n')

    # write list of arguments as seen by fortran.
    for a in args:
        # write delcartion of Fortran type
        o.write(a[i_arg_f_type])
        # attributes
        if a[i_arg_requirement] == 'optional':
            o.write(', optional')
        # pass function pointers by value
        if a[i_arg_type] == 'func':
            o.write(', value')
        else:
            o.write(', target')
        if a[i_arg_dims] != 'scalar':
            o.write(', %s'%a[i_arg_dims])

        o.write(", intent(%s) :: %s\n"%(a[i_arg_intent], a[i_arg_name]))

    # result type if this is a function
    if func_type != 'void':
        o.write(type_info[func_type]['f_type'] + ' :: res\n')

    if len(args) > 0:
        o.write('!\n')

    # declare auxiliary vaiables
    #   for string type we need to allocate new sotrage space and add tailing null character
    #   for bool type we need to convert it to C-type
    for a in args:
        t = a[i_arg_type]
        if t != 'func':
            o.write("type(C_PTR) :: %s_ptr\n"%a[i_arg_name])
        if t == 'string':
            o.write("%s, target, allocatable :: %s_c_type(:)\n"%(a[i_arg_c_type], a[i_arg_name]))
        if t == 'bool':
            o.write("%s, target :: %s_c_type\n"%(a[i_arg_c_type], a[i_arg_name]))

    if len(args) > 0:
        o.write('!\n')

    write_interface(o, func_name, func_type, args)

    # build a list of names for the C-interface call
    c_arg_names = []
    for a in args:
        t = a[i_arg_type]
        n = a[i_arg_name]
        if t == 'func':
            c_arg_names.append("%s"%n)
        else:
            c_arg_names.append("%s_ptr"%n)

    # before the call
    for a in args:
        t = a[i_arg_type]
        n = a[i_arg_name]

        # do nothing for function pointers
        if t == 'func':
            continue

        o.write("%s_ptr = C_NULL_PTR\n"%n)
        if a[i_arg_requirement] == 'optional':
            o.write("if (present(%s)) then\n"%n)

        if t == 'string':
            o.write("allocate(%s_c_type(len(%s)+1))\n"%(n, n))
            if a[i_arg_intent] in ('in', 'inout'):
                o.write("%s_c_type = string_f2c(%s)\n"%(n, n))
            o.write("%s_ptr = C_LOC(%s_c_type)\n"%(n, n))
        elif t == 'bool':
            if a[i_arg_intent] in ('in', 'inout'):
                o.write("%s_c_type = %s\n"%(n, n))
            o.write("%s_ptr = C_LOC(%s_c_type)\n"%(n, n))
        else:
            o.write("%s_ptr = C_LOC(%s)\n"%(n, n))

        if a[i_arg_requirement] == 'optional':
            o.write("endif\n")


    # make a function call
    string = 'call ' if func_type == 'void' else 'res = '
    string = string + func_name + '_aux(' + ','.join(c_arg_names) + ')'
    write_str_to_f90(o, string)

    # after the call
    for a in args:
        t = a[i_arg_type]
        n = a[i_arg_name]
        if t in ('bool', 'string'):

            if a[i_arg_requirement] == 'optional':
                o.write("if (present(%s)) then\n"%n)

            if t == 'string':
                if a[i_arg_intent] in ('inout', 'out'):
                    o.write("%s = string_c2f(%s_c_type)\n"%(n, n))
                o.write("deallocate(%s_c_type)\n"%n)
            elif t == 'bool':
                if a[i_arg_intent] in ('inout', 'out'):
                    o.write("%s = %s_c_type\n"%(n, n))
            else:
                pass

            if a[i_arg_requirement] == 'optional':
                o.write("endif\n")


    if func_type == 'void':
        o.write('end subroutine ')
    else:
        o.write('end function ')
    o.write(func_name + '\n\n')


def main():
    o = open('generated.f90', 'w')
    o.write('! Warning! This file is automatically generated from sirius_api.cpp using generate_api.py script!\n\n')
    o.write('!> @file generated.f90\n')
    o.write('!! @brief Autogenerated interface to Fortran.\n')

    for r in re.findall('(@api\s+begin\s+((\w|\W|\s)*?)@api\s+end)+', open(sys.argv[1], 'r').read()):
        di = yaml.safe_load(r[1])
        for func_name in di:
            o.write('!\n')
            func_type = di[func_name].get('return', 'void')
            func_doc = di[func_name].get('doc', '')
            details = di[func_name].get('full_doc', '')
            func_args = di[func_name].get('arguments', {})

            write_function(o, func_name, func_type, func_args, func_doc, details)


    o.close()

if __name__ == "__main__":
    main()
