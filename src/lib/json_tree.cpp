#include "json_tree.h"

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
        data = value.as_int();
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

template<> json_value_parser< std::vector<double> >::json_value_parser(const JSONNode& value, 
                                                                       std::vector<double>& data) : type_name_("vector<double>"), 
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

template<> json_value_parser<std::string>::json_value_parser(const JSONNode& value, 
                                                             std::string& data) : type_name_("string"), 
                                                                                  is_valid_(true)
{
    if (value.type() != JSON_STRING)
        is_valid_ = false;
        
    if (is_valid_) 
        data = value.as_string();
}


