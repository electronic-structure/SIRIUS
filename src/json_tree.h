// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file json_tree.h
 *   
 *  \brief Contains definition and implementation of JSON_value_parser, JSON_tree and JSON_write classes.
 */

#ifndef __JSON_TREE_H__
#define __JSON_TREE_H__

#include <sys/stat.h>
#include <libjson.h>
#include "utils.h"

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

/// JSON DOM tree.
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
            T val = T();
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
            T val = T();
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
    
        JSON_tree(JSONNode& node__, std::string& path__, const std::string& fname__) 
            : node_(node__), 
              path_(path__), 
              fname_(fname__)
        {
        }

        JSON_tree(std::string const& fname__) : fname_(fname__)
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

        inline bool exist(std::string const& name) const
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

        inline JSON_tree operator[](std::string const& key) const 
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

        inline JSON_tree operator[](int const key) const 
        {
            std::string new_path = path_ + std::string("/") + std::to_string(key);
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

/// Simple JSON serializer.
class JSON_write
{
    private:

        std::string fname_;
        
        FILE* fout_;

        int indent_step_;

        int indent_level_;

        bool new_block_;

        inline void new_indent_level(int shift)
        {
            indent_level_ += shift;
            new_block_ = true;
        }
        
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

    public:
        
        JSON_write(const std::string fname__) 
            : fname_(fname__), 
              indent_step_(4), 
              new_block_(true)
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

        inline void write(std::vector<double>& v)
        {
            new_line();
            fprintf(fout_, "[");
            for (int i = 0; i < (int)v.size(); i++)
            {
                if (i != 0) fprintf(fout_, ", ");
                fprintf(fout_, "%s", Utils::double_to_string(v[i]).c_str());
            }
            fprintf(fout_, "]");
        } 

        inline void write(std::string s)
        {
            new_line();
            fprintf(fout_, "\"%s\"", s.c_str());
        }

        inline void single(const char* name, int value)
        {
            new_line();
            fprintf(fout_, "\"%s\" : %i", name, value);
        }

        inline void single(const char* name, double value, int precision = -1)
        {
            new_line();
            std::string s = Utils::double_to_string(value, precision);
            fprintf(fout_, "\"%s\" : %s", name, s.c_str());
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
       
        /// Write array of doubles
        /** The following data structure is written:
         *  \code{.json}
         *      "name" : [v1, v2, v2, ...]
         *  \endcode
         */
        inline void single(const char* name, std::vector<double>& values)
        {
            new_line();
            fprintf(fout_, "\"%s\" : [", name);
            for (int i = 0; i < (int)values.size(); i++)
            {
                if (i) fprintf(fout_, ", ");
                std::string s = Utils::double_to_string(values[i]);
                fprintf(fout_, "%s", s.c_str());
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
        
        inline void single(const char* name, std::map<std::string, sirius::timer_stats> timers)
        {
            new_line();
            fprintf(fout_, "\"%s\" : {", name);
            
            indent_level_ += indent_step_;
            new_block_ = true;
            for (auto it = timers.begin(); it != timers.end(); it++)
            {
                std::vector<double> values(4); // total, min, max, average
                values[0] = it->second.total_value;
                values[1] = it->second.min_value;
                values[2] = it->second.max_value;
                values[3] = it->second.average_value;
                single(it->first.c_str(), values);
            }
            
            end_set();
        }

        inline void key(const char* name)
        {
            new_line();
            fprintf(fout_, "\"%s\" : ", name);
        }
            
        inline void begin_array(const char* name) // TODO: check for closed array wuth the same name
        {
            new_line();
            fprintf(fout_, "\"%s\" : [", name);
            
            new_indent_level(indent_step_);
        }

        inline void begin_array()
        {
            new_line();
            fprintf(fout_, "[");
            
            new_indent_level(indent_step_);
        }
        
        inline void end_array()
        {
            new_indent_level(-indent_step_);
            new_line();
            fprintf(fout_, "]");
        }
        
        inline void begin_set(const char* name)
        {
            new_line();
            fprintf(fout_, "\"%s\" : {", name);
            
            new_indent_level(indent_step_);
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
