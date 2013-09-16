#ifndef __JSON_TREE_H__
#define __JSON_TREE_H__

/// Auxiliary class for parsing JSONNode of the libjson
class JSON_value_parser 
{
    private:
        
        /// name of the type
        std::string type_name_;
        
        /// true if the node was parsed correctly
        bool is_valid_;
        
    public:
       
        JSON_value_parser(const JSONNode& value, bool& data) : type_name_("bool"), is_valid_(false)
        {
            if (value.type() == JSON_BOOL)
            {
                is_valid_ = true;
                data = value.as_bool();
            }
        }
        
        JSON_value_parser(const JSONNode& value, int& data) : type_name_("int"), is_valid_(false)
        {
            if (value.type() == JSON_NUMBER)
            {
                is_valid_ = true;
                data = (int)value.as_int();
            }
        }
        
        JSON_value_parser(const JSONNode& value, double& data) : type_name_("double"), is_valid_(false)
        {
            if (value.type() == JSON_NUMBER)
            {
                is_valid_ = true;
                data = value.as_float();
            }
        }
        
        JSON_value_parser(const JSONNode& value, std::vector<double>& data) : type_name_("vector<double>"), is_valid_(false)
        {
            if (value.type() == JSON_ARRAY) 
            {
                is_valid_ = true;
                data.clear();
                for (int i = 0; i < (int)value.size(); i++)
                {
                    double t;
                    JSON_value_parser v(value[i], t);
                    is_valid_ = is_valid_ && v.is_valid();
                    if (is_valid_) data.push_back(t);
                }
            }
        }
        
        JSON_value_parser(const JSONNode& value, std::vector<int>& data) : type_name_("vector<int>"), is_valid_(false)
        {
            if (value.type() == JSON_ARRAY) 
            {
                is_valid_ = true;
                data.clear();
                for (int i = 0; i < (int)value.size(); i++)
                {
                    int t;
                    JSON_value_parser v(value[i], t);
                    is_valid_ = is_valid_ && v.is_valid();
                    if (is_valid_) data.push_back(t);
                }
            }
        }

        JSON_value_parser(const JSONNode& value, std::string& data) : type_name_("string"), is_valid_(false)
        {
            if (value.type() == JSON_STRING) 
            {
                is_valid_ = true;
                data = value.as_string();
            }
        }
       
        /// Return name of the type
        const std::string& type_name()
        {
            return type_name_;
        }
        
        /// Return validity flag
        inline bool is_valid()
        {
            return is_valid_;
        }
};

class JSON_tree
{
    private:
        
        JSONNode node_;
        
        std::string path_;
        
        std::string fname_;
       
        /// Parse value or stop with the error
        template <typename T> 
        inline T parse_value()
        {
            T val;
            JSON_value_parser jvp(node_, val);
            if (!jvp.is_valid())
            {
                std::stringstream s;
                s << "null or invalid value of type " << jvp.type_name() << std::endl 
                  << "file : " << fname_ << std::endl
                  << "path : " << path_;
                error_local(__FILE__, __LINE__, s);
            }
            return val;
        }
        
        /// Parse value or return a default value
        template <typename T> 
        inline T parse_value(T& default_val)
        {
            T val;
            JSON_value_parser jvp(node_, val);
            if (!jvp.is_valid())
            {
                return default_val;
            }
            else
            {
                return val;
            }
        }
        
        void parse_file(const std::string& fname)
        {
            FILE* fin;
            if ((fin = fopen(fname.c_str(), "r")) == NULL)
            {
                std::stringstream s;
                s << "failed to open " << fname;
                error_local(__FILE__, __LINE__, s);
            }

            struct stat st;
            if (fstat(fileno(fin), &st) == 0)
            {
                std::string buffer(st.st_size + 1, 0);
                fread(&buffer[0], (int)buffer.size(), 1, fin);
                fclose(fin);
                node_ = libjson::parse(buffer);
            }
            else
            {
                error_local(__FILE__, __LINE__, "bad file handle");
            }
        }

    public:
    
        JSON_tree(JSONNode& node__, std::string& path__, const std::string& fname__) : 
            node_(node__), path_(path__), fname_(fname__)
        {
        }

        JSON_tree(const std::string& fname__) : fname_(fname__)
        {
            parse_file(fname_);
        }
        
        ~JSON_tree() 
        {
        }

        inline bool empty()
        {
            return node_.empty();
        }

        inline bool exist(const std::string& name) const
        {
            if (node_.find(name) == node_.end()) return false;
            return true;
        }

        inline int size() const
        {
            if (node_.type() == JSON_ARRAY || node_.type() ==  JSON_NODE)
            {
                return node_.size();
            }
            else
            {
                return 0;
            }
        }

        inline JSON_tree operator[](const char *key) const 
        {
            return (*this)[std::string(key)];
        }

        inline JSON_tree operator[](const std::string& key) const 
        {
            std::string new_path = path_ + std::string("/") + key;
            JSONNode n;
            try
            {
                n = node_.at(key);
            }
            catch (const std::out_of_range& e)
            {

            }
            return JSON_tree(n, new_path, fname_);
        }

        inline JSON_tree operator[](const int key) const 
        {
            std::string new_path = path_ + std::string("/") + Utils::to_string(key);
            JSONNode n;
            try
            {
                n = node_.at(key);
            }
            catch (const std::out_of_range& e)
            {

            }
            return JSON_tree(n, new_path, fname_);
        }
                
        /// Get a value or return a default value
        template <typename T> 
        inline T get(T default_val)
        {
            return parse_value(default_val);
        }
        
        /// Get a value or stop with an error
        template <typename T> 
        inline void operator>>(T& val)
        {
            val = parse_value<T>();
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
            if (fabs(value) > 1e-6)
            {
                fprintf(fout_, "\"%s\" : %.12f", name, value);
            }
            else
            {
                fprintf(fout_, "\"%s\" : %.12e", name, value);
            }
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
                if (fabs(values[i]) > 1e-6)
                {
                    fprintf(fout_, "%.12f", values[i]);
                }
                else
                {
                    fprintf(fout_, "%.12e", values[i]);
                }
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
