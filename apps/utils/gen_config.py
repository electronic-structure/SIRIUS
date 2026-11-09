'''
Generate the C++ accessors for the SIRIUS input parameters from input_schema.json.

Usage (from src/context):

    python3 ../../apps/utils/gen_config.py input_schema.json
    clang-format -i config.cpp

Two files are written to the current directory:
  - config.hpp: class declarations; only needs the nlohmann::json forward declarations
  - config.cpp: accessor definitions; includes the full nlohmann::json header

Keeping the definitions out of the header means that only config.cpp instantiates the nlohmann::json
templates needed by the accessors, instead of every translation unit that includes config.hpp.
'''
import json
import sys


def get_type(schema):
    '''
    Parse the node of the JSON schema and derive a C-type. If this is a native type or a (recursive) array of
    native types, return a pair (False, CType string). If array of objects was found, return a pair (True, None).
    Possible values of CType string can be: double, std::vector<int>, std::array<std::array<double, 3>, 6> etc.
    Complex numbers are not supported.
    '''
    json_to_cpp_type = {'string' : 'std::string', 'integer' : 'int', 'number' : 'double', 'boolean' : 'bool'}
    t = schema['type']
    if t != 'array':
        if t != 'object':
            return (False, json_to_cpp_type[t])
        else:
            return (True, None)
    else:
        format_str = "std::vector<%s>"
        if 'minItems' in schema and 'maxItems' in schema:
            if schema['minItems'] == schema['maxItems']:
                format_str = f"std::array<%s, {schema['minItems']}>"
        (is_array_of_objects, c_type) = get_type(schema['items'])
        if is_array_of_objects:
            return (True, None)
        else:
            return (False, format_str%c_type)


def doc_comment(schema):
    out = []
    if 'title' in schema:
        out.append(f'/// {schema["title"]}')
    if 'description' in schema:
        out.append('/**')
        for s in schema['description'].splitlines():
            out.append(f'    {s}')
        out.append('*/')
    return out


def get_object_as_cpp_class(object_name, schema, scope, defs):
    '''
    A special case for the array of objects. This objects are parsed here and two classes are created.
    First class describes the object itself, second class is introduced to query the size of the array
    and add extra elements.

    Declarations are returned; definitions are appended to defs. scope is the qualified name of the
    enclosing class (e.g. "config_t::hubbard_t::").
    '''
    output = []
    q = f'{scope}{object_name}_t::'

    output.append(f'class {object_name}_t')
    output.append('{')
    output.append('  private:')
    output.append('    nlohmann::json& dict_;')
    output.append('  public:')
    output.append(f'    {object_name}_t(nlohmann::json& dict__)')
    output.append('        : dict_(dict__)')
    output.append('    {')
    output.append('    }')
    for key in schema:
        (f1, ct1) = get_type(schema[key])
        output.append(f'    {ct1} {key}() const;')
        defs.append(f'auto {q}{key}() const -> {ct1}')
        defs.append('{')
        defs.append(f'    return dict_.at("{key}").get<{ct1}>();')
        defs.append('}')
        defs.append('')
    output.append(f'    bool contains(std::string key__) const;')
    defs.append(f'bool {q}contains(std::string key__) const')
    defs.append('{')
    defs.append('    return dict_.contains(key__);')
    defs.append('}')
    defs.append('')
    output.append('};')

    q = f'{scope}{object_name}_list_t::'
    output.append(f'class {object_name}_list_t')
    output.append('{')
    output.append('  private:')
    output.append('    nlohmann::json& dict_;')
    output.append('  public:')
    output.append(f'    {object_name}_list_t(nlohmann::json& dict__)')
    output.append('        : dict_(dict__)')
    output.append('    {')
    output.append('    }')
    output.append('    int size() const;')
    output.append('    void append(nlohmann::json& node__);')
    output.append('};')
    defs.append(f'int {q}size() const')
    defs.append('{')
    defs.append('    return dict_.size();')
    defs.append('}')
    defs.append('')
    defs.append(f'void {q}append(nlohmann::json& node__)')
    defs.append('{')
    defs.append('    dict_.push_back(node__);')
    defs.append('}')
    defs.append('')

    return output


def gen_input(object_name, path, schema, level, scope, defs):
    '''
    Generate the declaration of class {object_name}_t; definitions of its members are appended to defs.
    scope is the qualified name of the enclosing class ("" for the top level).
    '''
    # for this data srtucture: list of public and private members
    public_list = []
    private_list = []
    protected_list = []

    q = f'{scope}{object_name}_t::'

    if level == 0:
        # The dictionary is held by pointer so that nlohmann::json can stay an incomplete type in the header.
        # dict_ptr_ and dict_ must be declared before the sub-objects, which bind a reference to dict_.
        protected_list.append('std::unique_ptr<nlohmann::json> dict_ptr_;')
        protected_list.append('nlohmann::json& dict_;')
        public_list.append(f'{object_name}_t();')
        public_list.append(f'~{object_name}_t();')
        public_list.append(f'{object_name}_t({object_name}_t const&) = delete;')
        public_list.append(f'{object_name}_t& operator=({object_name}_t const&) = delete;')
        public_list.append('nlohmann::json const& dict() const')
        public_list.append('{')
        public_list.append('    return dict_;')
        public_list.append('}')
        defs.append(f'{q}{object_name}_t()')
        defs.append('    : dict_ptr_(std::make_unique<nlohmann::json>())')
        defs.append('    , dict_(*dict_ptr_)')
        defs.append('{')
        defs.append('}')
        defs.append('')
        defs.append(f'{q}~{object_name}_t() = default;')
        defs.append('')
    else:
        private_list.append('nlohmann::json& dict_;')
        # constructor of this object
        public_list.append(f'{object_name}_t(nlohmann::json& dict__)')
        public_list.append('    : dict_(dict__)')
        public_list.append('{')
        public_list.append('}')
    for key in schema:
        if schema[key]['type'] == 'object':
            public_list.extend(doc_comment(schema[key]))
            # traverse deeper into the data structure
            if 'properties' in schema[key]:
                data_t = gen_input(f'{key}', f'{path}/{key}', schema[key]['properties'], level + 1, q, defs)
                for e in data_t:
                    public_list.append(e)

                private_list.append(f'{key}_t {key}_{{dict_}};')
                public_list.append(f'inline auto const& {key}() const {{return {key}_;}}')
                public_list.append(f'inline auto& {key}() {{return {key}_;}}')
            # this is not a generic case
            if 'patternProperties' in schema[key]:
                # only ".*" is handeled
                # provide read-only access via label
                (is_array_of_objects, ct) = get_type(schema[key]['patternProperties']['.*'])
                if is_array_of_objects:
                    sys.exit(0)
                else:
                    public_list.append(f'{ct} {key}(std::string label__) const;')
                    defs.append(f'auto {q}{key}(std::string label__) const -> {ct}')
                    defs.append('{')
                    defs.append(f'    nlohmann::json::json_pointer p("{path}/{key}");')
                    defs.append(f'    return dict_.at(p / label__).get<{ct}>();')
                    defs.append('}')
                    defs.append('')
        else:
            # this is a simple type (not an object)
            (is_array_of_objects, ct) = get_type(schema[key])
            if is_array_of_objects:
                if 'title' in schema[key]:
                    public_list.append(f'/// {schema[key]["title"]}')
                out = get_object_as_cpp_class(key, schema[key]['items']['properties'], q, defs)
                for s in out:
                    public_list.append(s)
                public_list.append(f'{key}_t {key}(int idx__);')
                public_list.append(f'{key}_t {key}(int idx__) const;')
                public_list.append(f'{key}_list_t {key}();')
                public_list.append(f'{key}_list_t {key}() const;')
                for cv in ['', ' const']:
                    defs.append(f'auto {q}{key}(int idx__){cv} -> {key}_t')
                    defs.append('{')
                    defs.append(f'    nlohmann::json::json_pointer ptr("{path}/{key}");')
                    defs.append(f'    return {key}_t(dict_.at(ptr / idx__));')
                    defs.append('}')
                    defs.append('')
                for cv in ['', ' const']:
                    defs.append(f'auto {q}{key}(){cv} -> {key}_list_t')
                    defs.append('{')
                    defs.append(f'    nlohmann::json::json_pointer ptr("{path}/{key}");')
                    defs.append(f'    return {key}_list_t(dict_.at(ptr));')
                    defs.append('}')
                    defs.append('')
            else:
                public_list.extend(doc_comment(schema[key]))
                public_list.append(f'{ct} {key}() const;')
                public_list.append(f'void {key}({ct} {key}__);')
                defs.append(f'auto {q}{key}() const -> {ct}')
                defs.append('{')
                defs.append(f'    return dict_.at("{path}/{key}"_json_pointer).get<{ct}>();')
                defs.append('}')
                defs.append('')
                defs.append(f'void {q}{key}({ct} {key}__)')
                defs.append('{')
                defs.append('    if (dict_.contains("locked")) {')
                defs.append('        throw std::runtime_error(locked_msg);')
                defs.append('    }')
                defs.append(f'    dict_["{path}/{key}"_json_pointer] = {key}__;')
                defs.append('}')
                defs.append('')

    # compose a data type definition
    data_t = []

    data_t.append(f'class {object_name}_t')
    data_t.append('{')
    if protected_list:
        data_t.append('  protected:')
        for e in protected_list:
            data_t.append(f'    {e}')
    data_t.append('  public:')
    for e in public_list:
        data_t.append(f'    {e}')
    data_t.append('  private:')
    for e in private_list:
        data_t.append(f'    {e}')

    data_t.append('};')

    return data_t


schema_file = sys.argv[1] if len(sys.argv) > 1 else 'input_schema.json'
with open(schema_file) as f:
    data = json.load(f)

defs = []
data_t = gen_input('config', '', data['properties'], 0, '', defs)

warning = "// Warning! This file is autogenerated from input_schema.json using apps/utils/gen_config.py script"

with open('config.hpp', 'w') as f:
    f.write(f'{warning}\n\n')
    f.write('#ifndef __CONFIG_HPP__\n#define __CONFIG_HPP__\n\n')
    f.write('#include <array>\n#include <memory>\n#include <string>\n#include <vector>\n')
    f.write('#include "core/nlohmann_json_fwd.hpp"\n')
    f.write('\nnamespace sirius {\n\n')
    for e in data_t:
        f.write(f'{e}\n')
    f.write('\n} // namespace sirius\n\n#endif\n')

with open('config.cpp', 'w') as f:
    f.write(f'{warning}\n\n')
    f.write('#include <stdexcept>\n#include "context/config.hpp"\n#include "core/json.hpp"\n')
    f.write('\nnamespace sirius {\n\n')
    f.write('static std::string const locked_msg("parameters are locked");\n\n')
    for e in defs:
        f.write(f'{e}\n')
    f.write('} // namespace sirius\n')

with open('DOC.md', 'w') as f:
    for e in data['properties']:
        f.write(f'# {e}\n')
        f.write(f'{data["properties"][e]["title"]}\n')
        for k in data["properties"][e]["properties"]:
            f.write(f' - **{k}**: {data["properties"][e]["properties"][k]["title"]}\n')
        f.write('\n')
