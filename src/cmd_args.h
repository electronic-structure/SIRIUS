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

/** \file cmd_args.h
 *   
 *  \brief Contains definition and implementation of cmd_args class.
 */

#ifndef __CMD_ARGS_H__
#define __CMD_ARGS_H__

#include "vector3d.hpp"

/// Simple command line arguments handler.
class cmd_args
{
    private:
        
        std::vector< std::pair<std::string, std::string> > key_desc_;

        std::map<std::string, int> known_keys_;
        
        /// key->value mapping
        std::map<std::string, std::string> keys_;

    public:

        cmd_args()
        {
            register_key("--help", "print this help and exit");
        }

        void register_key(const std::string key__, const std::string description__)
        {
            key_desc_.push_back(std::pair<std::string, std::string>(key__, description__));
            
            int key_type = 0;
            std::string key = key__.substr(2, key__.length());
            
            if (key[key.length() - 1] == '=')
            {
                key = key.substr(0, key.length() - 1);
                key_type = 1;
            }

            if (known_keys_.count(key) != 0) TERMINATE("key is already added");

            known_keys_[key] = key_type;
        }

        void parse_args(int argn__, char** argv__)
        {
            for (int i = 1; i < argn__; i++)
            {
                std::string str(argv__[i]);
                if (str.length() < 3 || str[0] != '-' || str[1] != '-')
                {
                    std::stringstream s;
                    s << "wrong key: " << str;
                    TERMINATE(s);
                }

                size_t k = str.find("=");

                std::string key, val;
                if (k != std::string::npos)
                {
                    key = str.substr(2, k - 2);
                    val = str.substr(k + 1, str.length());
                }
                else
                {
                    key = str.substr(2, str.length());
                }

                if (known_keys_.count(key) != 1)
                {
                    std::stringstream s;
                    s << "key " << key << " is not found";
                    TERMINATE(s);
                }

                if (known_keys_[key] == 0 && k != std::string::npos)
                {
                    TERMINATE("this key must not have a value");
                }

                if (known_keys_[key] == 1 && k == std::string::npos)
                {
                    TERMINATE("this key must have a value");
                }

                if (keys_.count(key) != 0) TERMINATE("key is already added");

                keys_[key] = val;
            }
        }

        void print_help()
        {
            int max_key_width = 0;
            for (int i = 0; i < (int)key_desc_.size(); i++)
                max_key_width = std::max(max_key_width, (int)key_desc_[i].first.length());

            printf("Options:\n");

            for (int i = 0; i < (int)key_desc_.size(); i++)
            {
                printf("  %s", key_desc_[i].first.c_str());
                int k = (int)key_desc_[i].first.length();

                for (int j = 0; j < max_key_width - k + 1; j++) printf(" ");

                printf("%s\n", key_desc_[i].second.c_str());
            }
        }

        bool exist(const std::string key__) const
        {
            return keys_.count(key__);
        }

        template <typename T> 
        inline T value(const std::string key__) const;

        template <typename T> 
        inline T value(const std::string key__, T default_val__) const;

        std::string operator[](const std::string key__) const
        {
            return keys_.at(key__);
        }
};

template <>
inline int cmd_args::value<int>(const std::string key__) const
{
    int v;

    if (!exist(key__))
    {
        std::stringstream s;
        s << "command line parameter --" << key__ << " was not specified";
        TERMINATE(s);
    }

    std::istringstream(keys_.at(key__)) >> v;
    return v;
}

template <>
inline int cmd_args::value<int>(const std::string key__, int default_val__) const
{
    if (!exist(key__)) return default_val__;

    int v;
    std::istringstream(keys_.at(key__)) >> v;
    return v;
}

template <>
inline double cmd_args::value<double>(const std::string key__) const
{
    double v;

    if (!exist(key__))
    {
        std::stringstream s;
        s << "command line parameter --" << key__ << " was not specified";
        TERMINATE(s);
    }

    std::istringstream(keys_.at(key__)) >> v;
    return v;
}

template <>
inline double cmd_args::value<double>(const std::string key__, double default_val__) const
{
    if (!exist(key__)) return default_val__;

    double v;
    std::istringstream(keys_.at(key__)) >> v;
    return v;
}

template <>
inline std::string cmd_args::value<std::string>(const std::string key__) const
{
    return keys_.at(key__);
}

template <>
inline std::string cmd_args::value<std::string>(const std::string key__, const std::string default_val__) const
{
    if (!exist(key__)) return default_val__;
    return keys_.at(key__);
}

template <>
inline vector3d<double> cmd_args::value< vector3d<double> >(const std::string key__) const
{
    vector3d<double> v;

    if (!exist(key__))
    {
        std::stringstream s;
        s << "command line parameter --" << key__ << " was not specified";
        TERMINATE(s);
    }

    std::istringstream iss(keys_.at(key__));
    for (int x: {0, 1, 2}) {
        iss >> v[x];
    }
    return v;
}

template <>
inline vector3d<int> cmd_args::value< vector3d<int> >(const std::string key__) const
{
    vector3d<int> v;

    if (!exist(key__))
    {
        std::stringstream s;
        s << "command line parameter --" << key__ << " was not specified";
        TERMINATE(s);
    }

    std::istringstream iss(keys_.at(key__));
    for (int x: {0, 1, 2}) {
        iss >> v[x];
    }
    return v;
}

template<>
inline std::vector<int> cmd_args::value< std::vector<int> >(std::string const key__, std::vector<int> const default_val__) const
{
    if (!exist(key__)) return default_val__;

    std::istringstream iss(keys_.at(key__));
    std::vector<int> v;
    while (!iss.eof())
    {
        int k;
        iss >> k;
        v.push_back(k);
    }

    return v;
}

#endif // __CMD_ARGS_H__

