/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file cmd_args.cpp
 *
 *  \brief Contains definition on cmd_args class.
 *
 */

#include "core/cmd_args.hpp"

namespace sirius {

void
cmd_args::check_for_key(std::string const key__) const
{
    if (!exist(key__)) {
        std::stringstream s;
        s << "command line parameter --" << key__ << " was not specified";
        throw std::runtime_error(s.str());
    }
}

cmd_args::cmd_args()
{
    register_key("--help", "print this help and exit");
    parse_args(0, nullptr);
}

cmd_args::cmd_args(int argn__, char** argv__, std::initializer_list<std::pair<std::string, std::string>> keys__)
{
    register_key("--help", "print this help and exit");
    for (auto key : keys__) {
        register_key("--" + key.first, key.second);
    }
    parse_args(argn__, argv__);
}

void
cmd_args::register_key(std::string const key__, std::string const description__)
{
    key_desc_.push_back(std::pair<std::string, std::string>(key__, description__));

    int key_type    = 0;
    std::string key = key__.substr(2, key__.length());

    if (key[key.length() - 1] == '=') {
        key      = key.substr(0, key.length() - 1);
        key_type = 1;
    }

    if (known_keys_.count(key) != 0) {
        std::stringstream s;
        s << "key (" << key << ") is already registered";
        throw std::runtime_error(s.str());
    }

    known_keys_[key] = key_type;
}

void
cmd_args::parse_args(int argn__, char** argv__)
{
    for (int i = 1; i < argn__; i++) {
        std::string str(argv__[i]);
        if (str.length() < 3 || str[0] != '-' || str[1] != '-') {
            std::stringstream s;
            s << "wrong key: " << str;
            throw std::runtime_error(s.str());
        }

        size_t k = str.find("=");

        std::string key, val;
        if (k != std::string::npos) {
            key = str.substr(2, k - 2);
            val = str.substr(k + 1, str.length());
        } else {
            key = str.substr(2, str.length());
        }

        if (known_keys_.count(key) != 1) {
            std::stringstream s;
            s << "key " << key << " is not found";
            throw std::runtime_error(s.str());
        }

        if (known_keys_[key] == 0 && k != std::string::npos) {
            throw std::runtime_error("this key must not have a value");
        }

        if (known_keys_[key] == 1 && k == std::string::npos) {
            throw std::runtime_error("this key must have a value");
        }

        if (keys_.count(key) != 0) {
            std::stringstream s;
            s << "key (" << key << ") is already added";
            throw std::runtime_error(s.str());
        }

        keys_[key] = val;
    }
    if (this->exist("help")) {
        printf("Usage: %s [options]\n", argv__[0]);
        this->print_help();
        exit(0);
    }
}

void
cmd_args::print_help()
{
    int max_key_width = 0;
    for (int i = 0; i < (int)key_desc_.size(); i++) {
        max_key_width = std::max(max_key_width, (int)key_desc_[i].first.length());
    }

    std::printf("Options:\n");

    for (int i = 0; i < (int)key_desc_.size(); i++) {
        std::printf("  %s", key_desc_[i].first.c_str());
        int k = (int)key_desc_[i].first.length();

        for (int j = 0; j < max_key_width - k + 1; j++) {
            std::printf(" ");
        }

        std::printf("%s\n", key_desc_[i].second.c_str());
    }
}

} // namespace sirius
