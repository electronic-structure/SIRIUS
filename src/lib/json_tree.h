#ifndef __JSON_TREE_H__
#define __JSON_TREE_H__

#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "json.h"

template<typename T> class json_value_parser 
{
    private:
    
        std::string type_name_;
        
        bool is_valid_;
        
        bool is_empty_;
        
        T data_;
    
    public:
        
        json_value_parser(const Json::Value& value);
        
        inline T data()
        {
            return data_;
        }

        std::string& type_name()
        {
            return type_name_;
        }
        
        inline bool is_valid()
        {
            return is_valid_;
        }
        
        inline bool is_empty()
        {
            return is_empty_;
        }
};

template<> json_value_parser<int>::json_value_parser(const Json::Value& value) 
{
    type_name_ = "int";
    is_empty_ = value.empty();
    is_valid_ = value.isIntegral();
    if (is_valid_) 
        data_ = value.asInt();
}

template<> json_value_parser<double>::json_value_parser(const Json::Value& value) 
{
    type_name_ = "double";
    is_empty_ = value.empty();
    is_valid_ = value.isNumeric();
    if (is_valid_) 
        data_ = value.asDouble();
}

template<> json_value_parser< std::vector<double> >::json_value_parser(const Json::Value& value) 
{
    type_name_ = "vector<double>";
    is_empty_ = value.empty();
    is_valid_ = value.isArray();
    if (is_valid_) 
    {
        data_.clear();
        for (int i = 0; i < (int)value.size(); i++) 
        {
            json_value_parser<double> t(value[i]);
            is_empty_ = is_empty_ && t.is_empty();
            is_valid_ = is_valid_ && t.is_valid();
            if (is_valid_) 
                data_.push_back(t.data());
        }
    }
}

template<> json_value_parser<std::string>::json_value_parser(const Json::Value& value) 
{
    type_name_ = "string";
    is_empty_ = value.empty();
    is_valid_ = value.isString();
    if (is_valid_) 
        data_ = value.asString();
}

class JsonTree 
{
    private:
        
        Json::Value node;
        
        std::string path;
        
        template <typename T> inline bool parse_value(T& val)
        {
            json_value_parser<T> v(node);
            if (v.is_empty() || (!v.is_valid())) 
                return false;
            else
            {
                val = v.data();
                return true;
            }

        }
        
    public:
    
        JsonTree() : node(0)  
        {
        }

        JsonTree(Json::Value node, 
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
            Json::Reader reader;
            Json::Value root;
            std::ifstream in(fname.c_str());
            
            if (!reader.parse(in, root)) 
            {
                stop(std::cout << "Error parsing " << fname << std::endl << 
                     reader.getFormatedErrorMessages() << std::endl);
            }
            node = root;
        }

        inline int size() 
        {
            return node.size();
        }
    
        inline JsonTree operator [] (const char *key) const 
        {
            std::string new_path = path + std::string("/") + std::string(key);
            return JsonTree(node[key], new_path);
        }
    
        inline JsonTree operator [] (const int key) const 
        {
            std::stringstream s;
            s << key;
            std::string new_path = path + std::string("/") + s.str();
            return JsonTree(node[(Json::UInt)key], new_path);
        }

        template <typename T> inline T get()
        {
            T val;
            if (parse_value(val)) 
                return val;
            else
                stop(std::cout << "null or invalid value" << std::endl << "path : " << path);
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
                stop(std::cout << "null or invalid value" << std::endl << "path : " << path);
        }
        
};

#endif // __JSON_TREE_H__
