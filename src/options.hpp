#ifndef __OPTIONS_HPP__
#define __OPTIONS_HPP__

struct param {
    enum kind {kind_int, kind_double, kind_vecint, kind_vecdouble, kind_bool, kind_vecbool, kind_string, kind_vecstring};
    kind type_;
    alignas (void*) char data[256];
    std::string section_;
    std::string name_;
    std::string use_;
    std::string desc_;
    bool do_not_destroy{false};
    param() {
    };

    param (const param &src__)
        {
            section_ = src__.section_;
            type_ = src__.type_;
            name_ = src__.name_;
            use_ = src__.use_;
            desc_ = src__.desc_;
            memcpy(data, src__.data, 256);
        }

    param (param &&src__)
        {
            section_ = src__.section_;
            type_ = src__.type_;
            name_ = src__.name_;
            use_ = src__.use_;
            desc_ = src__.desc_;
            memcpy(data, src__.data, 256);
            src__.do_not_destroy = true;
        }

    void init(std::string section__, std::string name__, std::string use__, std::string desc__, int value) {
        const char* p = reinterpret_cast<const char*>(&value);
        // the only safe way to copy is one byte at a time using char* (to not violate strict aliasing rules)
        std::copy(p, p+sizeof(int), data);
        type_ = kind_int;
        section_ = section__;
        name_ = name__;
        use_ = use__;
        desc_ = desc__;
    }

    void init(std::string section__, std::string name__, std::string use__, std::string desc__, double value) {
        const char* p = reinterpret_cast<const char*>(&value);
        std::copy(p, p+sizeof(double), data);
        type_ = kind_double;
        section_ = section__;
        name_ = name__;
        use_ = use__;
        desc_ = desc__;
    }

    void init(std::string section__, std::string name__, std::string use__, std::string desc__, bool value) {
        const char* p = reinterpret_cast<const char*>(&value);
        std::copy(p, p+sizeof(bool), data);
        section_ = section__;
        type_ = kind_bool;
        name_ = name__;
        use_ = use__;
        desc_ = desc__;
    }

    void init(std::string section__, std::string name__, std::string use__, std::string desc__, std::string value) {
        const char* p = reinterpret_cast<const char*>(&value);
        std::copy(p, p+sizeof(std::string), data);
        section_ = section__;
        name_ = name__;
        use_ = use__;
        desc_ = desc__;
        type_ = kind_string;
    }

    void init(std::string section__, std::string name__, std::string use__, std::string desc__, const std::vector<int>& value) {
        void* p = reinterpret_cast<void*>(data);
        new(p) std::vector<int>(value); // construct in place.
        section_ = section__;
        name_ = name__;
        use_ = use__;
        desc_ = desc__;
        type_ = kind_vecint;
    }

    void init(std::string section__, std::string name__, std::string use__, std::string desc__, const std::vector<std::string>& value) {
        void* p = reinterpret_cast<std::string*>(data);
        new(p) std::vector<std::string>(value); // construct in place.
        section_ = section__;
        name_ = name__;
        use_ = use__;
        desc_ = desc__;
        type_ = kind_vecstring;
    }

    void init(std::string section__, std::string name__, std::string use__, std::string desc__, const std::vector<double>& value) {
        void* p = reinterpret_cast<void*>(data);
        new(p) std::vector<double>(value); // construct in place.
        section_ = section__;
        name_ = name__;
        use_ = use__;
        desc_ = desc__;
        type_ = kind_vecdouble;
    }

    void init(std::string section__, std::string name__, std::string use__, std::string desc__, const std::vector<bool>& value) {
        void* p = reinterpret_cast<void*>(data);
        new(p) std::vector<bool>(value); // construct in place.
        section_ = section__;
        name_ = name__;
        use_ = use__;
        desc_ = desc__;
        type_ = kind_vecbool;
    }

    int get_int() const {
        if (type_ != kind_int) throw std::runtime_error("not an int");
        return *reinterpret_cast<const int*>(data);
    }

    std::vector<int> get_int_vec() const {
        if (type_ != kind_vecint) throw std::runtime_error("not an int vector");
        return *reinterpret_cast<const std::vector<int>*>(data);
    }

    double get_double() const {
        if (type_ != kind_double) throw std::runtime_error("not a double");
        return *reinterpret_cast<const double*>(data);
    }

    std::vector<double> get_double_vec() const {
        if (type_ != kind_vecint) throw std::runtime_error("not an int vector");
        return *reinterpret_cast<const std::vector<double>*>(data);
    }

    bool get_bool() const {
        if (type_ != kind_bool) throw std::runtime_error("not a boolean");
        return *reinterpret_cast<const bool*>(data);
    }

    std::string get_string() const {
        if (type_ != kind_string) throw std::runtime_error("not a boolean");
        return *reinterpret_cast<const std::string*>(data);
    }

    std::vector<bool> get_bool_vec() const {
        if (type_ != kind_vecbool) throw std::runtime_error("not an boolean vector");
        return *reinterpret_cast<const std::vector<bool>*>(data);
    }

    std::vector<std::string> get_string_vec() const {
        if (type_ != kind_vecstring) throw std::runtime_error("not an boolean vector");
        return *reinterpret_cast<const std::vector<std::string>*>(data);
    }
    std::string get_usage() const {
        return use_;
    }

    std::string get_name() const {
        return name_;
    }

    std::string get_description() const {
        return desc_;
    }

    int get_type() const {
        if (type_ == kind_int) {
            return 0;
        }
        if (type_ == kind_double) {
            return 1;
        }
        if (type_ == kind_bool) {
            return 2;
        }
        if (type_ == kind_string) {
            return 3;
        }
        if (type_ == kind_vecint) {
            return 5;
        }
        if (type_ == kind_vecdouble) {
            return 6;
        }
        if (type_ == kind_vecbool) {
            return 7;
        }
        if (type_ == kind_vecstring) {
            return 8;
        }
        throw std::runtime_error("undefined type_");
    }

    int get_length() const {
        if (type_ == kind_vecint) {
            return get_int_vec().size();
        }
        if (type_ == kind_vecdouble) {
            return get_double_vec().size();
        }
        if (type_ == kind_vecbool) {
            return get_bool_vec().size();
        }
        if (type_ == kind_vecstring) {
            return get_string_vec().size();
        }
        return 1;
    }

    // Destructor has to manually call the destructor of objects that use heap storage.
    ~param() {
        if (do_not_destroy)
            return;
        if (type_ == kind_vecint) {
            using T=std::vector<int>;
            T* p = reinterpret_cast<T*>(data);
            p->~T();
        }
        if (type_ == kind_vecdouble) {
            using T=std::vector<double>;
            T* p = reinterpret_cast<T*>(data);
            p->~T();
        }
        if (type_ == kind_vecdouble) {
            using T=std::vector<bool>;
            T* p = reinterpret_cast<T*>(data);
            p->~T();
        }
        if (type_ == kind_vecstring) {
            using T=std::vector<bool>;
            T* p = reinterpret_cast<T*>(data);
            p->~T();
        }
    }
};



// int main() {
//     param pi(23);
//     param pd(23.4);
//     param pv(std::vector<int>{1, 2, 3, 5, 7, 11});

//     std::cout << pi.get_int() << "\n";
//     std::cout << pd.get_double() << "\n";
//     for (auto x: pv.get_int_vec()) std::cout << x << " "; std::cout << "\n";

//     try {
//         auto x = pv.get_int();
//     }
//     catch (std::runtime_error e) {
//         std::cout << "caught an exception, like we expected! ... " << e.what() << "\n";
//     }
// }


// struct option_value_ {
//     char *name_;
//     char *desc_;
//     char *usage_;
//     enum { real64, integer, boolean, str
//     } type_;
//     int length_{1};
//     double dval_[16];
//     double ival_[16];
//     bool bval_[16];
//     char *cval_[16];

//     add(char *name__, char *desc__, char *usage__, double dval__)
//         {
//             name_ = name__;
//             desc_ = desc__;
//             usage_ = usage__;
//             dval_[0] = dval__;
//         }
//     add(char *name__, char *desc__, char *usage__, double *dval__, int length__);
//         {
//             name_ = name__;
//             desc_ = desc__;
//             usage_ = usage__;
//             if (length__ < 16)
//                 length_ = length__;
//             for (int i = 0; i < length__; i++)
//                 dval_[i] = dval__[i];
//         }
//     add(char *name__, char *desc__, char *usage__, int *ival__, int length__);
//         {
//             name_ = name__;
//             desc_ = desc__;
//             usage_ = usage__;
//             if (length__ < 16)
//                 length_ = length__;
//             for (int i = 0; i < length__; i++)
//                 ival_[i] = ival__[i];
//         }
//     add(char *name__, char *desc__, char *usage__, bool *ival__, int length__);
//         {
//             name_ = name__;
//             desc_ = desc__;
//             usage_ = usage__;
//             if (length__ < 16)
//                 length_ = length__;
//             for (int i = 0; i < length__; i++)
//                 bval_[i] = bval__[i];
//         }
//     add(char *name__, char *desc__, char *usage__, char **ival__, int length__);
//         {
//             name_ = name__;
//             desc_ = desc__;
//             usage_ = usage__;
//             if (length__ < 16)
//                 length_ = length__;
//             for (int i = 0; i < length__; i++)
//                 strcpy(bval_[i], bval__[i]);
//         }
//     add(char *name__, char *desc__, char *usage__, char *ival__, int length__);
//         {
//             name_ = name__;
//             desc_ = desc__;
//             usage_ = usage__;
//             strcpy(bval_[0], bval__);
//         }
// };

#endif
