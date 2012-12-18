#ifndef __JSON_TREE_H__
#define __JSON_TREE_H__

template<typename T> class json_value_parser 
{
    private:
    
        std::string type_name_;
        
        bool is_valid_;
        
    public:
        
        json_value_parser(const JSONNode& value, T& val);
        
        const std::string& type_name()
        {
            return type_name_;
        }
        
        inline bool is_valid()
        {
            return is_valid_;
        }
};

template<> json_value_parser<bool>::json_value_parser(const JSONNode& value, 
                                                      bool& data) : type_name_("bool"), 
                                                                    is_valid_(true)
{
    if (value.type() != JSON_BOOL) 
        is_valid_ = false;
    
    if (is_valid_) 
        data = value.as_bool();
}


template<> json_value_parser<int>::json_value_parser(const JSONNode& value, 
                                                     int& data) : type_name_("int"), 
                                                                  is_valid_(true)
{
    if (value.type() != JSON_NUMBER) 
        is_valid_ = false;
    
    if (is_valid_) 
        data = (int)value.as_int();
}

template<> json_value_parser<double>::json_value_parser(const JSONNode& value, 
                                                        double& data) : type_name_("double"), 
                                                                        is_valid_(true)
{
    if (value.type() != JSON_NUMBER)
        is_valid_ = false;
        
    if (is_valid_) 
        data = value.as_float();
}

template<> json_value_parser< std::vector<double> >::json_value_parser
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
            json_value_parser<double> v(value[i], t);
            is_valid_ = is_valid_ && v.is_valid();
            if (is_valid_) 
                data.push_back(t);
        }
    }
}

template<> json_value_parser< std::vector<int> >::json_value_parser
    (const JSONNode& value, std::vector<int>& data) : type_name_("vector<int>"), 
                                                      is_valid_(true)
{
    if (value.type() != JSON_ARRAY) 
        is_valid_ = false;
    
    if (is_valid_) 
    {
        data.clear();
        for (int i = 0; i < (int)value.size(); i++)
        {
            int t;
            json_value_parser<int> v(value[i], t);
            is_valid_ = is_valid_ && v.is_valid();
            if (is_valid_) 
                data.push_back(t);
        }
    }
}

template<> json_value_parser<std::string>::json_value_parser
    (const JSONNode& value, std::string& data) : type_name_("string"), 
                                                 is_valid_(true)
{
    if (value.type() != JSON_STRING)
        is_valid_ = false;
        
    if (is_valid_) 
        data = value.as_string();
}

class JsonTree 
{
    private:
        
        JSONNode node;
        
        std::string path;
        
        std::string fname_;
        
        template <typename T> inline bool parse_value(T& val, std::string& type_name)
        {
            json_value_parser<T> jvp(node, val);
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
            std::string key(key_);
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
                
        template <typename T> inline T get(T& default_val)
        {
            T val;
            std::string type_name;
            
            if (parse_value(val, type_name)) 
                return val;
            else
                return default_val;
        }
        
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

#endif // __JSON_TREE_H__
