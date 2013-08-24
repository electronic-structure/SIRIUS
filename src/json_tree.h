#ifndef __JSON_TREE_H__
#define __JSON_TREE_H__

template<typename T> 
class JSON_value_parser 
{
    private:
    
        std::string type_name_;
        
        bool is_valid_;
        
    public:
        
        JSON_value_parser(const JSONNode& value, T& val);
        
        const std::string& type_name()
        {
            return type_name_;
        }
        
        inline bool is_valid()
        {
            return is_valid_;
        }
};

template<> JSON_value_parser<bool>::JSON_value_parser(const JSONNode& value, 
                                                      bool& data) : type_name_("bool"), 
                                                                    is_valid_(true)
{
    if (value.type() != JSON_BOOL) 
        is_valid_ = false;
    
    if (is_valid_) 
        data = value.as_bool();
}


template<> JSON_value_parser<int>::JSON_value_parser(const JSONNode& value, 
                                                     int& data) : type_name_("int"), 
                                                                  is_valid_(true)
{
    if (value.type() != JSON_NUMBER) 
        is_valid_ = false;
    
    if (is_valid_) 
        data = (int)value.as_int();
}

template<> JSON_value_parser<double>::JSON_value_parser(const JSONNode& value, 
                                                        double& data) : type_name_("double"), 
                                                                        is_valid_(true)
{
    if (value.type() != JSON_NUMBER)
        is_valid_ = false;
        
    if (is_valid_) 
        data = value.as_float();
}

template<> JSON_value_parser< std::vector<double> >::JSON_value_parser
    (const JSONNode& value, std::vector<double>& data) : type_name_("vector<double>"), 
                                                         is_valid_(true)
{
    if (value.type() != JSON_ARRAY) 
        is_valid_ = false;
    
    if (is_valid_) 
    {
        data.clear();
        for (int i = 0; i < (int)value.size(); i++)
        {
            double t;
            JSON_value_parser<double> v(value[i], t);
            is_valid_ = is_valid_ && v.is_valid();
            if (is_valid_) 
                data.push_back(t);
        }
    }
}

template<> JSON_value_parser< std::vector<int> >::JSON_value_parser
    (const JSONNode& value, std::vector<int>& data) : type_name_("vector<int>"), 
                                                      is_valid_(true)
{
    if (value.type() != JSON_ARRAY) is_valid_ = false;
    
    if (is_valid_) 
    {
        data.clear();
        for (int i = 0; i < (int)value.size(); i++)
        {
            int t;
            JSON_value_parser<int> v(value[i], t);
            is_valid_ = is_valid_ && v.is_valid();
            if (is_valid_) data.push_back(t);
        }
    }
}

template<> JSON_value_parser<std::string>::JSON_value_parser
    (const JSONNode& value, std::string& data) : type_name_("string"), 
                                                 is_valid_(true)
{
    if (value.type() != JSON_STRING) is_valid_ = false;
        
    if (is_valid_) data = value.as_string();
}

class JsonTree 
{
    private:
        
        JSONNode node;
        
        std::string path;
        
        std::string fname_;
        
        template <typename T> inline bool parse_value(T& val, std::string& type_name)
        {
            JSON_value_parser<T> jvp(node, val);
            type_name = jvp.type_name();
            return jvp.is_valid();
        }
        
    public:
    

        JsonTree(JSONNode& node, 
                 std::string& path,
                 const std::string& fname_) : node(node),
                                              path(path),
                                              fname_(fname_)
        {
        }

        JsonTree(const std::string& fname) : fname_(fname)
        {
            parse(fname);
        }
        
        ~JsonTree() 
        {
        };

        inline bool empty()
        {
            return node.empty();
        }

        inline bool exist(const std::string& name)
        {
            if (node.find(name) == node.end()) return false;
            return true;
        }
    
        void parse(const std::string& fname)
        {
            std::ifstream ifs;
            ifs.open(fname.c_str(), std::ios::binary);
            if (ifs.fail())
            {
                std::stringstream s;
                s << "fail to open " << fname;
                error(__FILE__, __LINE__, s, fatal_err);
            }
            ifs.seekg(0, std::ios::end);
            std::streamoff length = ifs.tellg();
            ifs.seekg(0, std::ios::beg);

            std::string buffer(length, ' ');
            ifs.read(&buffer[0], length);
            ifs.close();

            node = libjson::parse(buffer);
        }

        inline int size() 
        {
            if ((node.type() == JSON_ARRAY) || (node.type() ==  JSON_NODE))
                return node.size();
            else
                return 0;
        }

        inline JsonTree operator [] (const char *key_) const 
        {
            return (*this)[std::string(key_)];
            /*std::string key(key_);
            std::string new_path = path + std::string("/") + key;
            JSONNode n;
            try
            {
                n = node.at(key);
            }
            catch (const std::out_of_range& e)
            {

            }
            return JsonTree(n, new_path, fname_);*/
        }

        inline JsonTree operator [] (const std::string& key) const 
        {
            std::string new_path = path + std::string("/") + key;
            JSONNode n;
            try
            {
                n = node.at(key);
            }
            catch (const std::out_of_range& e)
            {

            }
            return JsonTree(n, new_path, fname_);
        }

        inline JsonTree operator [] (const int key) const 
        {
            std::stringstream s;
            s << key;
            std::string new_path = path + std::string("/") + s.str();
            JSONNode n;
            try
            {
                n = node.at(key);
            }
            catch (const std::out_of_range& e)
            {

            }
            return JsonTree(n, new_path, fname_);
        }

        template <typename T> inline T get()
        {
            T val;
            std::string type_name;
            
            if (!parse_value(val, type_name))
            {
                std::stringstream s;
                s << "null or invalid value of type " << type_name << std::endl 
                  << "file : " << fname_ << std::endl
                  << "path : " << path;
                error(__FILE__, __LINE__, s, fatal_err);
            }

            return val;
        }
                
        /*template <typename T> inline T get(T& default_val)
        {
            T val;
            std::string type_name;
            
            if (parse_value(val, type_name)) 
                return val;
            else
                return default_val;
        }*/
        
        template <typename T> inline T get(T default_val)
        {
            T val;
            std::string type_name;
            
            if (parse_value(val, type_name)) 
                return val;
            else
                return default_val;
        }
        
        template <typename T> inline void operator >> (T& val)
        {
            std::string type_name;
            
            if (!parse_value(val, type_name))
            { 
                std::stringstream s;
                s << "null or invalid value of type " << type_name << std::endl 
                  << "file : " << fname_ << std::endl
                  << "path : " << path;
                error(__FILE__, __LINE__, s, fatal_err);
            }
        }
}; 

class JSON_write
{
    private:

        std::string fname_;
        
        FILE* fout_;

        int indent_step_;

        int indent_level_;

        bool new_block_;

        inline void new_line()
        {
            std::string s(indent_level_, ' ');
            if (new_block_)
            {
                fprintf(fout_, "\n%s", s.c_str());
                new_block_ = false;
            }
            else
            {
                fprintf(fout_, ",\n%s", s.c_str());
            }
        }

        inline void new_indent_level(int shift)
        {
            indent_level_ += shift;
            new_block_ = true;
        }

    public:
        
        JSON_write(const std::string fname__) : fname_(fname__), indent_step_(4), new_block_(true)
        {
            fout_ = fopen(fname_.c_str(), "w");
            fprintf(fout_, "{");
            indent_level_ = indent_step_;
        }

        ~JSON_write()
        {
            fprintf(fout_, "\n}\n");
            fclose(fout_);
        }

        inline void single(const char* name, int value)
        {
            new_line();
            fprintf(fout_, "\"%s\" : %i", name, value);
        }

        inline void single(const char* name, double value)
        {
            new_line();
            fprintf(fout_, "\"%s\" : %.12f", name, value);
        }

        inline void single(const char* name, const std::string& value)
        {
            new_line();
            fprintf(fout_, "\"%s\" : \"%s\"", name, value.c_str());
        }

        inline void string(const char* name, const std::string& value)
        {
            new_line();
            fprintf(fout_, "\"%s\" : %s", name, value.c_str());
        }
        
        inline void single(const char* name, std::vector<double>& values)
        {
            new_line();
            fprintf(fout_, "\"%s\" : [", name);
            for (int i = 0; i < (int)values.size(); i++)
            {
                if (i) fprintf(fout_, ",");
                fprintf(fout_, "%.12f", values[i]);
            }
            fprintf(fout_, "]");
        }
        
        inline void single(const char* name, std::vector<int>& values)
        {
            new_line();
            fprintf(fout_, "\"%s\" : [", name);
            for (int i = 0; i < (int)values.size(); i++)
            {
                if (i) fprintf(fout_, ",");
                fprintf(fout_, "%i", values[i]);
            }
            fprintf(fout_, "]");
        }
        
        inline void single(const char* name, std::vector<std::string>& values)
        {
            new_line();
            fprintf(fout_, "\"%s\" : [", name);
            for (int i = 0; i < (int)values.size(); i++)
            {
                if (i) fprintf(fout_, ",");
                fprintf(fout_, "\"%s\"", values[i].c_str());
            }
            fprintf(fout_, "]");
        }
        
        inline void single(const char* name, std::map<std::string, sirius::timer_descriptor*>& timer_descriptors)
        {
            new_line();
            fprintf(fout_, "\"%s\" : {", name);
            
            indent_level_ += indent_step_;
            std::map<std::string, sirius::timer_descriptor*>::iterator it;
            new_block_ = true;
            for (it = timer_descriptors.begin(); it != timer_descriptors.end(); it++)
            {
                std::vector<double> tv(2);
                tv[0] = it->second->total;
                tv[1] = (it->second->count == 0) ? 0.0 : it->second->total / it->second->count;
                single(it->first.c_str(), tv);
            }
            
            end_set();
        }

        inline void begin_array(const char* name)
        {
            new_line();
            fprintf(fout_, "\"%s\" : [", name);
            
            new_indent_level(indent_step_);
        }
        
        inline void end_array()
        {
            new_indent_level(-indent_step_);
            new_line();
            fprintf(fout_, "]");
        }
        
        inline void begin_set()
        {
            new_line();
            fprintf(fout_, "{");
            
            new_indent_level(indent_step_);
        }
        
        inline void end_set()
        {
            new_indent_level(-indent_step_);
            new_line();
            fprintf(fout_, "}");
        }
};



#endif // __JSON_TREE_H__
