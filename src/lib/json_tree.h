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
        
        template <typename T> inline bool parse_value(T& val)
        {
            json_value_parser<T> v(node, val);
            return v.is_valid();
        }
        
    public:
    

        JsonTree(JSONNode& node, 
                 std::string& path) : node(node),
                                      path(path)  
        {
        }

        JsonTree(const std::string& fname) 
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
                std::cout << "error opening " << fname << std::endl;
                exit(0);
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
            return JsonTree(n, new_path);
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
            return JsonTree(n, new_path);
        }

        template <typename T> inline T get()
        {
            T val;
            if (parse_value(val)) 
                return val;
            else
            {
                std::cout << "null or invalid value" << std::endl << "path : " << path << std::endl;
                exit(0);
            }
        }
                
        template <typename T> inline T get(T& default_val)
        {
            T val;
            if (parse_value(val)) 
                return val;
            else
                return default_val;
        }
        
        template <typename T> inline T get(T default_val)
        {
            T val;
            if (parse_value(val)) 
                return val;
            else
                return default_val;
        }
        
        template <typename T> inline void operator >> (T& val)
        {
            if (!parse_value(val))
            { 
                std::cout << "null or invalid value" << std::endl << "path : " << path << std::endl;
                exit(0);
            }
        }
}; 

#endif // __JSON_TREE_H__
