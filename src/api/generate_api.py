#!/usr/bin/env python

import argparse
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
    },
    'gs_handler' : {
        'f_type' : 'type(sirius_ground_state_handler)',
        'c_type' : 'type(C_PTR)'
    },
    'ks_handler' : {
        'f_type' : 'type(sirius_kpoint_set_handler)',
        'c_type' : 'type(C_PTR)'
    },
    'md_handler' : {
        'f_type' : 'type(sirius_md_extrapolation)',
        'c_type' : 'type(C_PTR)'
    },
    'ctx_handler' : {
        'f_type' : 'type(sirius_context_handler)',
        'c_type' : 'type(C_PTR)'
    }
}

class ArgumentAttribute:
    def __init__(self, attr_str):
        idx = attr_str.find('dimension')
        if idx >= 0:
            idx1 = attr_str.find('(', idx)
            idx2 = attr_str.find(')', idx1)
            if idx1 != -1 and idx2 != -1 and idx2 > idx1:
                self.dimension_ = attr_str[idx1:idx2+1]
                attr_str = attr_str[:idx] + attr_str[idx2+1:]
            else:
                raise Exception(f'wrong attribute string: {attr_str}')
        else:
            self.dimension_ = 'scalar'

        attr = attr_str.replace(' ', '').split(',')
        valid_attr = ['', 'in', 'out', 'inout', 'optional', 'required', 'value', 'ptr']
        # default is inout
        self.intent_ = 'inout'
        # required by default
        self.required_ = 'required'
        # pass by pointer by default
        self.pass_by_ = 'ptr'
        for a in attr:
            if a not in valid_attr:
                raise Exception(f'wrong attribute: {a}')
            if a in ['in', 'out', 'inout']:
                self.intent_ = a
            if a in ['required', 'optional']:
                self.required_ = a
            if a in ['value', 'ptr']:
                self.pass_by_ = a

    def dimension(self):
        return self.dimension_

    def intent(self):
        return self.intent_

    def required(self):
        return self.required_

    def pass_by(self):
        return self.pass_by_


class Argument:
    def __init__(self, name, arg_dict):
        self.attr_ = ArgumentAttribute(arg_dict['attr'])
        self.type_id_ = arg_dict['type']
        self.doc_ = arg_dict['doc']
        self.name_ = name
        if self.type_id_ == 'func' and not (self.attr_.intent() == 'in' and
            self.attr_.required() == 'required' and self.attr_.pass_by() == 'value'):
            raise Exception("wroing attributes for function pointer")
        if self.attr_.intent() in ['out', 'inout'] and self.attr_.pass_by() == 'value':
            raise Exception("wroing 'value' attribute for output variables")
        if self.attr_.required() == 'optional' and self.attr_.pass_by() == 'value':
            raise Exception("can't pass optional argument by value")
        if self.type_id() == 'string' and self.attr_.pass_by() == 'value':
            raise Exception("can't pass string by value")

    def name(self):
        return self.name_

    def attr(self):
        return self.attr_

    def type_id(self):
        return self.type_id_

    def c_type(self):
        return type_info[self.type_id()]['c_type']

    def f_type(self):
        return type_info[self.type_id()]['f_type']

    def doc(self):
        return self.doc_

    def write_fortran_api_arg(self, out):
        # write delcartion of Fortran type
        out.write(self.f_type())
        # attributes
        if self.attr().required() == 'optional':
            out.write(', optional')
        # pass function pointers by value
        if self.attr().pass_by() == 'value':
            out.write(', value')
        else:
            out.write(', target')
        #if self.attr().dimension() != 'scalar':
        #    out.write(f', {self.attr().dimension()}')

        out.write(f', intent({self.attr().intent()}) :: {self.name()}')
        if self.attr().dimension() != 'scalar':
            out.write(f'{self.attr().dimension()}')
        out.write('\n')

    def write_interface_api_arg(self, out):
        if self.type_id() == 'func':
            out.write(f'type(C_FUNPTR), value :: {self.name()}\n')
        else:
            if self.attr().pass_by() == 'value':
                out.write(f'{self.c_type()}, value :: {self.name()}\n')
            else:
                out.write(f'type(C_PTR), value :: {self.name()}\n')

    def write_tmp_vars(self, out):
        if self.attr().pass_by() == 'ptr':
            out.write(f'type(C_PTR) :: {self.name()}_ptr\n')
        if self.type_id() == 'string':
            out.write(f'{self.c_type()}, target, allocatable :: {self.name()}_c_type(:)\n')
        if self.type_id() == 'bool':
            out.write(f'{self.c_type()}, target :: {self.name()}_c_type\n')

    def prepend_interface_call(self, out):

        # do nothing for function pointers
        if self.type_id() == 'func':
            return

        if self.attr().pass_by() == 'ptr':
            out.write(f'{self.name()}_ptr = C_NULL_PTR\n')
            if self.attr().required() == 'optional':
                out.write(f'if (present({self.name()})) then\n')

        if self.type_id() == 'string':
            out.write(f'allocate({self.name()}_c_type(len({self.name()})+1))\n')
            if self.attr().intent() in ('in', 'inout'):
                out.write(f'{self.name()}_c_type = string_f2c({self.name()})\n')
            if self.attr().pass_by() == 'ptr':
                out.write(f'{self.name()}_ptr = C_LOC({self.name()}_c_type)\n')
        elif self.type_id() == 'bool':
            if self.attr().intent() in ('in', 'inout'):
                out.write(f'{self.name()}_c_type = {self.name()}\n')
            if self.attr().pass_by() == 'ptr':
                out.write(f'{self.name()}_ptr = C_LOC({self.name()}_c_type)\n')
        elif self.type_id() in ['gs_handler', 'ks_handler', 'ctx_handler']:
            if self.attr().pass_by() == 'ptr':
                out.write(f'{self.name()}_ptr = C_LOC({self.name()}%handler_ptr_)\n')
        else:
            if self.attr().pass_by() == 'ptr':
                out.write(f'{self.name()}_ptr = C_LOC({self.name()})\n')

        if self.attr().pass_by() == 'ptr' and self.attr().required() == 'optional':
            out.write("endif\n")

    def append_interface_call(self, out):

        if self.type_id() in ('bool', 'string'):

            if self.attr().required() == 'optional':
                out.write(f'if (present({self.name()})) then\n')

            if self.type_id() == 'string':
                if self.attr().intent() in ['inout', 'out']:
                    out.write(f'{self.name()} = string_c2f({self.name()}_c_type)\n')
                out.write(f'deallocate({self.name()}_c_type)\n')
            elif self.type_id() == 'bool':
                if self.attr().intent() in ['inout', 'out']:
                    out.write(f'{self.name()} = {self.name()}_c_type\n')
            else:
                pass

            if self.attr().required() == 'optional':
                out.write("endif\n")


def write_str_to_f90(o, string):
    p = 0
    while (True):
        p = string.find(',', p)
        # no more commas left in the string or string is short
        if p == -1 or len(string) <= 80:
            o.write("%s\n"%string)
            break
        # position after comma
        p += 1
        if p >= 80:
            o.write("%s&\n&"%string[:p])
            string = string[p:]
            p = 0


def write_function_doc(o, func_doc, details, func_args):
    '''
    Write a comment block with function documentation
    '''
    # write documentation header
    o.write(f'!> @brief {func_doc}\n')
    if details:
        o.write('!> @details\n')
        for l in details.split('\n'):
            o.write(f'!> {l}\n')
        #for l in details:
        #    o.write("!> %s\n"%l)

    for a in func_args:
        o.write(f'!> @param [{a.attr().intent()}] {a.name()} {a.doc()}\n')

def write_interface(o, func_name, func_type, args):
    '''
    Write interface block
    '''

    # create interface section
    o.write('interface\n')

    string = 'subroutine ' if func_type == 'void' else 'function '
    string = string + func_name + '_aux(' + ','.join([a.name() for a in args]) + ')'

    if (func_type == 'void'):
        string = string + '&'
    else:
        string = string + ' result(res)&'
    write_str_to_f90(o, string)
    o.write("&bind(C, name=\"%s\")\n"%func_name)

    o.write('use, intrinsic :: ISO_C_BINDING\n')
    # declare arguments of the interface (as seen by C code)
    for a in args:
        a.write_interface_api_arg(o)

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
    args = [Argument(a, func_args[a]) for a in func_args] if func_args else []

    # write documentation header
    write_function_doc(o, func_doc, details, args)

    # write declaration of function or subroutine
    string = 'subroutine ' if func_type == 'void' else 'function '
    string = string + func_name + '(' + ','.join([a.name() for a in args]) + ')'
    if func_type != 'void':
        string = string + ' result(res)'
    write_str_to_f90(o, string)
    o.write('implicit none\n')
    o.write('!\n')

    # write list of arguments as seen by fortran.
    for a in args:
        a.write_fortran_api_arg(o)

    # result type if this is a function
    if func_type != 'void':
        o.write(type_info[func_type]['f_type'] + ' :: res\n')

    if len(args) > 0:
        o.write('!\n')

    # declare auxiliary vaiables
    #   for string type we need to allocate new storage space and add tailing null character
    #   for bool type we need to convert it to C-type
    for a in args:
        a.write_tmp_vars(o)

    if len(args) > 0:
        o.write('!\n')

    write_interface(o, func_name, func_type, args)

    # build a list of names for the C-interface call
    c_arg_names = []
    for a in args:
        if a.attr().pass_by() == 'value':
            c_arg_names.append(f'{a.name()}')
        else:
            c_arg_names.append(f'{a.name()}_ptr')

    # before the call
    for a in args:
        a.prepend_interface_call(o)

    # make a function call
    string = 'call ' if func_type == 'void' else 'res = '
    string = string + func_name + '_aux(' + ','.join(c_arg_names) + ')'
    write_str_to_f90(o, string)

    # after the call
    for a in args:
        a.append_interface_call(o)

    if func_type == 'void':
        o.write('end subroutine ')
    else:
        o.write('end function ')
    o.write(func_name + '\n\n')


def main():
    parser = argparse.ArgumentParser(
            description='Generate the Fortran API from the given FILE (usually the sirius_api.cpp)',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("FILE", type=argparse.FileType('r'))
    parser.add_argument('-o', '--output', type=argparse.FileType('w'),
                        default='sirius.f90', help='The output file name')
    args = parser.parse_args()

    args.output.write(f'! Warning! This file is automatically generated from {args.FILE.name} using the generate_api.py script!\n')
    #args.output.write('!> @file generated.f90\n')
    #args.output.write('!! @brief Autogenerated interface to Fortran.\n')

    text1='''
!> @file sirius.f90
!! @brief Autogenerated Fortran module for the SIRIUS API.
module sirius

use, intrinsic :: ISO_C_BINDING

INTEGER, PARAMETER, PUBLIC :: SIRIUS_INTEGER_TYPE = 1
INTEGER, PARAMETER, PUBLIC :: SIRIUS_LOGICAL_TYPE = 2
INTEGER, PARAMETER, PUBLIC :: SIRIUS_STRING_TYPE = 3
INTEGER, PARAMETER, PUBLIC :: SIRIUS_NUMBER_TYPE = 4
INTEGER, PARAMETER, PUBLIC :: SIRIUS_OBJECT_TYPE = 5
INTEGER, PARAMETER, PUBLIC :: SIRIUS_ARRAY_TYPE = 6

INTEGER, PARAMETER, PUBLIC :: SIRIUS_INTEGER_ARRAY_TYPE = 7
INTEGER, PARAMETER, PUBLIC :: SIRIUS_LOGICAL_ARRAY_TYPE = 8
INTEGER, PARAMETER, PUBLIC :: SIRIUS_NUMBER_ARRAY_TYPE = 9
INTEGER, PARAMETER, PUBLIC :: SIRIUS_STRING_ARRAY_TYPE = 10
INTEGER, PARAMETER, PUBLIC :: SIRIUS_OBJECT_ARRAY_TYPE = 11
INTEGER, PARAMETER, PUBLIC :: SIRIUS_ARRAY_ARRAY_TYPE = 12

!> @brief Opaque wrapper for simulation context handler.
type sirius_context_handler
    type(C_PTR) :: handler_ptr_
end type

!> @brief Opaque wrapper for DFT ground statee handler.
type sirius_ground_state_handler
    type(C_PTR) :: handler_ptr_
end type

!> @brief Opaque wrapper for K-point set handler.
type sirius_kpoint_set_handler
    type(C_PTR) :: handler_ptr_
end type

!> @brief Opaque wrapper for K-point set handler.
type sirius_md_extrapolation
    type(C_PTR) :: handler_ptr_
end type



!> @brief Free any of the SIRIUS handlers (context, ground state or k-points).
interface sirius_free_handler
    module procedure sirius_free_handler_ctx, sirius_free_handler_ks, sirius_free_handler_dft
end interface

contains

!> Internal function that adds trailing null character to the string to make it C-style.
function string_f2c(f_string) result(res)
    implicit none
    character(kind=C_CHAR,len=*), intent(in)  :: f_string
    character(kind=C_CHAR,len=1) :: res(len_trim(f_string) + 1)
    integer i
    do i = 1, len_trim(f_string)
        res(i) = f_string(i:i)
    end do
    res(len_trim(f_string) + 1) = C_NULL_CHAR
end function string_f2c

!> Internal function that converts C-string (with trailing null character) to the Fortran string.
function string_c2f(c_string) result(res)
    implicit none
    character(kind=C_CHAR,len=1), intent(in) :: c_string(:)
    character(kind=C_CHAR,len=size(c_string) - 1) :: res
    character(C_CHAR) c
    integer i
    do i = 1, size(c_string)
        c = c_string(i)
        if (c == C_NULL_CHAR) then
            res(i:) = ' '
            exit
        endif
        res(i:i) = c
    end do
end function string_c2f
'''

    text2='''
subroutine sirius_free_handler_ctx(handler, error_code)
    implicit none
    type(sirius_context_handler), intent(inout) :: handler
    integer, optional, target, intent(out) :: error_code
    call sirius_free_object_handler(handler%handler_ptr_, error_code)
end subroutine sirius_free_handler_ctx

subroutine sirius_free_handler_ks(handler, error_code)
    implicit none
    type(sirius_kpoint_set_handler), intent(inout) :: handler
    integer, optional, target, intent(out) :: error_code
    call sirius_free_object_handler(handler%handler_ptr_, error_code)
end subroutine sirius_free_handler_ks

subroutine sirius_free_handler_dft(handler, error_code)
    implicit none
    type(sirius_ground_state_handler), intent(inout) :: handler
    integer, optional, target, intent(out) :: error_code
    call sirius_free_object_handler(handler%handler_ptr_, error_code)
end subroutine sirius_free_handler_dft

end module
'''
    args.output.write(text1)

    print(f'Writing autogenerated Fortran interface to: {args.output.name}', file=sys.stderr)

    nfunc = 0
    for r in re.findall(r'(@api\s+begin\s+((\w|\W|\s)*?)@api\s+end)+', args.FILE.read()):
        di = yaml.safe_load(r[1])
        for func_name in di:
            args.output.write('!\n')
            func_type = di[func_name].get('return', 'void')
            func_doc = di[func_name].get('doc', '')
            details = di[func_name].get('full_doc', '')
            func_args = di[func_name].get('arguments', {})

            flg = False
            for a in func_args:
                if a == 'error_code':
                    flg = True
            if not flg:
                print(f"function {func_name} doesn't have the error_code parameter")

            write_function(args.output, func_name, func_type, func_args, func_doc, details)
            nfunc = nfunc + 1

    print(f"Total number of API functions : {nfunc}")

    args.output.write(text2)

if __name__ == "__main__":
    main()
