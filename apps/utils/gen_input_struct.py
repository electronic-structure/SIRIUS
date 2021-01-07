import json

json_to_cpp_type = {'string' : 'std::string', 'integer' : 'int', 'number' : 'double', 'boolean' : 'bool'}

def gen_input(namespace, path, schema, level):
    # for this level: list of public and private members
    public_list = []
    private_list = []
    if level == 0:
        private_list.append('nlohmann::json dict_;')
    else:
        private_list.append('nlohmann::json const& dict_;')
        public_list.append(f'{namespace}_t(nlohmann::json const& dict__)')
        public_list.append('    : dict_(dict__)')
        public_list.append('{')
        public_list.append('}')
    for key in schema:
        if schema[key]['type'] == 'object':
            # traverse deeper into the data structure
            data_t = gen_input(f'{key}', f'{path}/{key}', schema[key]['properties'], level + 1)
            for e in data_t:
                public_list.append(e)

            private_list.append(f'{key}_t {key}_{{dict_}};')
            public_list.append(f'inline auto {key}() const {{return {key}_;}}')
        else:
            # this is a simple type
            t = schema[key]['type']
            public_list.append(f'inline auto {key}() const {{return dict_["{path}/{key}"_json_pointer].get<{json_to_cpp_type[t]}>();}}')

    # compose a data type definition
    data_t = []

    data_t.append(f'class {namespace}_t')
    data_t.append('{')
    data_t.append('  public:')
    for e in public_list:
        data_t.append(f'    {e}')
    data_t.append('  private:')
    for e in private_list:
        data_t.append(f'    {e}')
    data_t.append('};')

    return data_t



    #indent = ' ' * level
    #if level != 0:
    #    indent2 = ' ' * (level - 2)
    #    print(f'{indent2}public:')
    #print(f'{indent}class {namespace}_t {{')
    #indent = ' ' * (level + 4)
    #indent2 = ' ' * (level + 2)
    #if level == 0:
    #    print(f'{indent2}private:')
    #    print(f'{indent}nlohmann::json dict_;')
    #else:
    #    print(f'{indent2}private:')
    #    print(f'{indent}nlohmann::json const& dict_;')
    #    print(f'{indent2}public:')
    #    print(f'{indent}{namespace}_t(nlohmann::json const& dict__)')
    #    print(f'{indent}    : dict_(dict__)')
    #    print(f'{indent}{{')
    #    print(f'{indent}}}')
    #for key in schema:
    #    if schema[key]['type'] == 'object':
    #        # traverse deeper into the data structure
    #        gen_input(f'{key}', f'{path}/{key}', schema[key]['properties'], level + 4)
    #        print(f'{indent2}private:')
    #        print(f'{indent}{key}_t {key}_{{dict_}};')
    #        print(f'{indent2}public:')
    #        print(f'{indent}inline auto {key}() const {{return {key}_;}}')
    #    else:
    #        # this is a simple type
    #        t = schema[key]['type']
    #        print(f'{indent2}public:')
    #        print(f'{indent}inline auto {key}() const {{return dict_["{path}/{key}"_json_pointer].get<{json_to_cpp_type[t]}>();}}')

    #indent = ' ' * level
    #print(f'{indent}}};')
    #return


with open('input_schema.json') as f:
  data = json.load(f)

header = '''

#include <string>
#include "json.hpp"

'''

footer = '''
int main(int argn, char** argv)
{
    nlohmann::json dict;
    input_t in;
    return 0;
}
'''

print(header)

data_t = gen_input('input', '', data['properties'], 0)
for e in data_t:
    print(e)

print(footer)

