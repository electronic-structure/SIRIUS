#ifndef __JSON_TREE_H__
#define __JSON_TREE_H__

#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "../libjson/libjson.h"
#include "error_handling.h"

template<typename T> class json_value_parser 
{
    private:
    
        std::string type_name_;
        
        bool is_valid_;
        
    public:
        
        json_value_parser(const JSONNode& value, 
                          T& val);
        
        const std::string& type_name()
        {
            return type_name_;
        }
        
        inline bool is_valid()
        {
            return is_valid_;
        }
};

class JsonTree 
{
    private:
        
        JSONNode node;
        
        std::string path;
        
        std::string fname_;
        
        template <typename T> inline bool parse_value(T& val, std::string& type_name)
        {
            json_value_parser<T> v(node, val);
            type_name = v.type_name();
            return v.is_valid();
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
                error(__FILE__, __LINE__, s.str().c_str());
            }
            ifs.seekg(0, std::ios::end);
            int length = ifs.tellg();
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
                error(__FILE__, __LINE__, s.str().c_str());
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
                error(__FILE__, __LINE__, s.str().c_str());
            }
        }
}; 

#endif // __JSON_TREE_H__
