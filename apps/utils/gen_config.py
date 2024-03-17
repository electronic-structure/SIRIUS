import json

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


def get_object_as_cpp_class(object_name, schema):
    '''
    A special case for the array of objects. This objects are parsed here and two classes are created.
    First class describes the object itself, second class is introduced to query the size of the array
    and add extra elements.
    '''
    output = []
    
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
        output.append(f'    auto {key}() const')
        output.append('    {')
        output.append(f'        return dict_.at("{key}").get<{ct1}>();')
        output.append('    }')
    output.append(f'    bool contains(std::string key__) const')
    output.append('    {')
    output.append('        return dict_.contains(key__);')
    output.append('    }')
   
    output.append('};')
    output.append(f'class {object_name}_list_t')
    output.append('{')
    output.append('  private:')
    output.append('    nlohmann::json& dict_;')
    output.append('  public:')
    output.append(f'    {object_name}_list_t(nlohmann::json& dict__)')
    output.append('        : dict_(dict__)')
    output.append('    {')
    output.append('    }')
    output.append('    int size() const')
    output.append('    {')
    output.append('        return dict_.size();')
    output.append('    }')
    output.append('    void append(nlohmann::json& node__)')
    output.append('    {')
    output.append('        dict_.push_back(node__);')
    output.append('    }')
    output.append('};')

    return output


def gen_input(object_name, path, schema, level):
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
        # constructor of this object
        public_list.append(f'{object_name}_t(nlohmann::json& dict__)')
        public_list.append('    : dict_(dict__)')
        public_list.append('{')
        public_list.append('}')
    for key in schema:
        if schema[key]['type'] == 'object':
            if 'title' in schema[key]:
                public_list.append(f'/// {schema[key]["title"]}')
            if 'description' in schema[key]:
                public_list.append('/**')
                for s in schema[key]['description'].splitlines():
                    public_list.append(f'    {s}')
                public_list.append('*/')
            # traverse deeper into the data structure
            if 'properties' in schema[key]:
                data_t = gen_input(f'{key}', f'{path}/{key}', schema[key]['properties'], level + 1)
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
                    public_list.append(f'inline auto {key}(std::string label__) const')
                    public_list.append('{')
                    public_list.append(f'    nlohmann::json::json_pointer p("{path}/{key}");')
                    public_list.append(f'    return dict_.at(p / label__).get<{ct}>();')
                    public_list.append('}')
        else:
            # this is a simple type (not an object)
            (is_array_of_objects, ct) = get_type(schema[key])
            if is_array_of_objects:
                out = get_object_as_cpp_class(key, schema[key]['items']['properties'])
                if 'title' in schema[key]:
                    public_list.append(f'/// {schema[key]["title"]}')
                for s in out:
                    public_list.append(s)
                public_list.append(f'{key}_t {key}(int idx__)')
                public_list.append('{')
                public_list.append(f'    nlohmann::json::json_pointer ptr("{path}/{key}");')
                public_list.append(f'    return {key}_t(dict_.at(ptr / idx__));')
                public_list.append('}')
                public_list.append(f'{key}_t {key}(int idx__) const')
                public_list.append('{')
                public_list.append(f'    nlohmann::json::json_pointer ptr("{path}/{key}");')
                public_list.append(f'    return {key}_t(dict_.at(ptr / idx__));')
                public_list.append('}')
                public_list.append(f'{key}_list_t {key}()')
                public_list.append('{')
                public_list.append(f'    nlohmann::json::json_pointer ptr("{path}/{key}");')
                public_list.append(f'    return {key}_list_t(dict_.at(ptr));')
                public_list.append('}')
                public_list.append(f'{key}_list_t {key}() const')
                public_list.append('{')
                public_list.append(f'    nlohmann::json::json_pointer ptr("{path}/{key}");')
                public_list.append(f'    return {key}_list_t(dict_.at(ptr));')
                public_list.append('}')


            else:
                if 'title' in schema[key]:
                    public_list.append(f'/// {schema[key]["title"]}')
                if 'description' in schema[key]:
                    public_list.append('/**')
                    for s in schema[key]['description'].splitlines():
                        public_list.append(f'    {s}')
                    public_list.append('*/')
                public_list.append(f'inline auto {key}() const')
                public_list.append('{')
                public_list.append(f'    return dict_.at("{path}/{key}"_json_pointer).get<{ct}>();')
                public_list.append('}')
                public_list.append(f'inline void {key}({ct} {key}__)')
                public_list.append('{')
                public_list.append('    if (dict_.contains("locked")) {')
                public_list.append('        throw std::runtime_error(locked_msg);')
                public_list.append('    }')
                public_list.append(f'    dict_["{path}/{key}"_json_pointer] = {key}__;')
                public_list.append('}')

    # compose a data type definition
    data_t = []

    data_t.append(f'class {object_name}_t')
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

print("// Warning! This file is autogenerated from input_schema.json using gen_input_struct.py script")
print("\nnamespace sirius {\n")
print("std::string const locked_msg(\"parameters are locked\");\n")
data_t = gen_input('config', '', data['properties'], 0)
for e in data_t:
    print(e)

print("\n}")

with open('DOC.md', 'w') as f:
    for e in data['properties']:
        f.write(f'# {e}\n')
        f.write(f'{data["properties"][e]["title"]}\n')
        for k in data["properties"][e]["properties"]:
            f.write(f' - **{k}**: {data["properties"][e]["properties"][k]["title"]}\n')
        f.write('\n')
