
class cmd_args
{
    private:
        
        std::vector< std::pair<std::string, std::string> > key_desc_;

        std::map<std::string, int> known_keys_;
        
        /// key->value mapping
        std::map<std::string, std::string> keys_;

    public:

        void register_key(const std::string key__, const std::string description)
        {
            key_desc_.push_back(std::pair<std::string, std::string>(key__, description));
            
            int key_type = 0;
            std::string key = key__.substr(2, key__.length());
            
            if (key[key.length() - 1] == '=')
            {
                key = key.substr(0, key.length() - 1);
                key_type = 1;
            }

            if (known_keys_.count(key) != 0) terminate(__FILE__, __LINE__, "key is already added");

            known_keys_[key] = key_type;
        }

        void parse_args(int argn, char** argv)
        {
            for (int i = 1; i < argn; i++)
            {
                std::string str(argv[i]);
                if (str.length() < 3 || str[0] != '-' || str[1] != '-') terminate(__FILE__, __LINE__, "wrong key");

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
                    terminate(__FILE__, __LINE__, s);
                }

                if (known_keys_[key] == 0 && k != std::string::npos)
                {
                    terminate(__FILE__, __LINE__, "this key must not have a value");
                }

                if (known_keys_[key] == 1 && k == std::string::npos)
                {
                    terminate(__FILE__, __LINE__, "this key must have a value");
                }

                if (keys_.count(key) != 0) terminate(__FILE__, __LINE__, "key is already added");

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

        bool exist(const std::string key)
        {
            return keys_.count(key);
        }

        template <typename T> 
        T value(const std::string key);
};

template <>
int cmd_args::value<int>(const std::string key)
{
    int v;

    if (!exist(key))
    {
        std::stringstream s;
        s << "command line parameter --" << key << " was not specified";
        terminate(__FILE__, __LINE__, s);
    }

    std::istringstream(keys_[key]) >> v;
    return v;
}

