import json

json_to_cpp_type = {'string' : 'std::string', 'integer' : 'int', 'number' : 'double', 'boolean' : 'bool'}

def gen_input(namespace, path, schema, level):
    # for this data srtucture: list of public and private members
    public_list = []
    private_list = []
    protected_list = []
    if level == 0:
        protected_list.append('nlohmann::json dict_;')
        public_list.append('nlohmann::json const& dict() const')
        public_list.append('{')
        public_list.append('    return dict_;')
        public_list.append('}')
    else:
        private_list.append('nlohmann::json& dict_;')
        public_list.append(f'{namespace}_t(nlohmann::json& dict__)')
        public_list.append('    : dict_(dict__)')
        public_list.append('{')
        public_list.append('}')
    for key in schema:
        if schema[key]['type'] == 'object':
            if 'description' in schema[key]:
                public_list.append(f'/// {schema[key]["description"]}')
            # traverse deeper into the data structure
            data_t = gen_input(f'{key}', f'{path}/{key}', schema[key]['properties'], level + 1)
            for e in data_t:
                public_list.append(e)

            private_list.append(f'{key}_t {key}_{{dict_}};')
            public_list.append(f'inline auto const& {key}() const {{return {key}_;}}')
            public_list.append(f'inline auto& {key}() {{return {key}_;}}')
        else:
            # this is a simple type
            t = schema[key]['type']
            ct = json_to_cpp_type[t]

            if 'description' in schema[key]:
                public_list.append(f'/// {schema[key]["description"]}')
            public_list.append(f'inline auto {key}() const')
            public_list.append('{')
            public_list.append(f'    return dict_["{path}/{key}"_json_pointer].get<{ct}>();')
            public_list.append('}')
            public_list.append(f'inline void {key}({ct} {key}__)')
            public_list.append('{')
            public_list.append(f'    dict_["{path}/{key}"_json_pointer] = {key}__;')
            public_list.append('}')

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
    if protected_list:
        data_t.append('  protected:')
        for e in protected_list:
            data_t.append(f'    {e}')

    data_t.append('};')

    return data_t


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
    config_t in;
    return 0;
}
'''

print(header)

data_t = gen_input('config', '', data['properties'], 0)
for e in data_t:
    print(e)

print(footer)

