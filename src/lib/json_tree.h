#ifndef __JSON_TREE_H__
#define __JSON_TREE_H__

#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "json.h"

enum json_error_handling {ignore, warn, die};

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

class json_tree 
{
    private:
        
        Json::Value node;
        
        json_error_handling on_invalid_value;
         
        json_error_handling on_empty_value;
        
    public:
    
        json_tree() : node(0)  
        {
            on_invalid_value = die;
            on_empty_value = warn;
        }

        json_tree(Json::Value node, 
                  json_error_handling on_invalid_value = die, 
                  json_error_handling on_empty_value = warn) : node(node), 
                                                               on_invalid_value(on_invalid_value), 
                                                               on_empty_value(on_empty_value) 
        {
        }

        json_tree(const std::string& fname) : on_invalid_value(die),
                                              on_empty_value(warn)
        {
            parse(fname);
        }
        
        ~json_tree() 
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
    
        inline json_tree operator [] (const char *key) const 
        {
            return json_tree(node[key], on_invalid_value, on_empty_value);
        }
    
        inline json_tree operator [] (const int key) const 
        {
            return json_tree(node[(Json::UInt)key], on_invalid_value, on_empty_value);
        }

        template <typename T> inline void operator >> (T &var)
        {
            json_value_parser<T> v(node);
            
            if (v.is_empty()) 
            {
                std::string err_msg = "null value of type " + v.type_name();
                switch(on_empty_value)
                {
                    case ignore:
                        break;
                        
                    case warn:
                        throw(std::logic_error(err_msg));
                    
                    case die:
                        stop(std::cout << err_msg);

                }
            }

            if (!v.is_valid()) 
            {
                std::string err_msg = "invalid value of type " + v.type_name();
                switch(on_invalid_value)
                {
                    case ignore:
                        break;
                        
                    case warn:
                        throw(std::logic_error(err_msg));

                    case die:
                        stop(std::cout << err_msg);
                }
            }

            var = v.data();
        }
};

#endif // __JSON_TREE_H__
