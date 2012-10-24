
namespace sirius
{

class hdf5_tree
{
    private:

        class hdf5_group
        {
            public:
                
                hid_t id;
                
                hdf5_group(hid_t file_id, const std::string path)
                {
                    id = H5Gopen(file_id, path.c_str(), H5P_DEFAULT);
                    if (id < 0)
                        error(__FILE__, __LINE__, "error in H5Gopen()");
                }

                ~hdf5_group()
                {            
                    if (H5Gclose(id) < 0)
                        error(__FILE__, __LINE__, "error in H5Gclose()");
                }
        };

        class hdf5_dataspace
        {
            public:
                
                hid_t id;

                hdf5_dataspace(const std::vector<int> dims)
                {
                    std::vector<hsize_t> current_dims(dims.size());
                    for (int i = 0; i < (int)dims.size(); i++)
                        current_dims[dims.size() - i - 1] = dims[i];

                    id = H5Screate_simple(dims.size(), &current_dims[0], NULL);
                    
                    if (id < 0)
                        error(__FILE__, __LINE__, "error in H5Screate_simple()");
                }

                ~hdf5_dataspace()
                {
                    if (H5Sclose(id) < 0)
                        error(__FILE__, __LINE__, "error in H5Sclose()");
                }
        };

        class hdf5_dataset
        {
            public:

                hid_t id;
                
                hdf5_dataset(hid_t group_id, const std::string& name)
                {
                    id = H5Dopen(group_id, name.c_str(), H5P_DEFAULT);
                    if (id < 0)
                        error(__FILE__, __LINE__, "error in H5Dopen()");
                }

                ~hdf5_dataset()
                {
                    if (H5Dclose(id) < 0)
                        error(__FILE__, __LINE__, "error in H5Dclose()");
                }
        };


        std::string file_name_;
        std::string path_;
        
        hid_t file_id_;
        
        bool root_node_;
        
    public:

        bool static file_exists(const std::string& file_name__)
        {
            std::ifstream ifs(file_name__.c_str());
            if (ifs.is_open()) return true;
            return false;
        }
    
        hdf5_tree(const std::string& file_name__, bool truncate = false) : file_name_(file_name__), 
                                                                           file_id_(-1),
                                                                           root_node_(true)
        {
            if (H5open() < 0)
                error(__FILE__, __LINE__, "error in H5open()");
            
            if (hdf5_trace_errors)
                H5Eset_auto(H5E_DEFAULT, NULL, NULL);
            
            if (truncate)
            {
                // create a new file
                file_id_ = H5Fcreate(file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                if (file_id_ < 0)
                    error(__FILE__, __LINE__, "error in H5Fcreate()");
            }
            else
            {
                if (file_exists(file_name_))
                {
                    // try to open existing file
                    file_id_ = H5Fopen(file_name_.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

                    if (file_id_ < 0)
                        error(__FILE__, __LINE__, "H5Fopen() failed");
                }
                else
                {
                    // create a new file if it doesn't exist
                    file_id_ = H5Fcreate(file_name_.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);

                    if (file_id_ < 0)
                        error(__FILE__, __LINE__, "error in H5Fcreate()");
                }
            }

            path_ = "/";
        }

        hdf5_tree(hid_t file_id__, const std::string& path__) : path_(path__),
                                                                file_id_(file_id__),
                                                                root_node_(false)
        {
        }

        ~hdf5_tree()
        {
            if (root_node_)
                if (H5Fclose(file_id_) < 0)
                    error(__FILE__, __LINE__, "error in H5Fclose()");
        }

        hdf5_tree create_node(const std::string& name)
        {
            // try to open a group
            hdf5_group group(file_id_, path_);
            
            // try to create a new group
            hid_t new_group_id = H5Gcreate(group.id, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (new_group_id < 0)
            {
                std::stringstream s;
                s << "error in H5Gcreate()" << std::endl
                  << "path : " << path_ << std::endl
                  << "name : " << name;
                error(__FILE__, __LINE__, s);
            }
            else if (H5Gclose(new_group_id) < 0)
                error(__FILE__, __LINE__, "error in H5Gclose()");
                
            return (*this)[name];
        }
        
        template <typename T>
        void write(const std::string& name, T* data, const std::vector<int>& dims)
        {
            // open group
            hdf5_group group(file_id_, path_);

            hdf5_dataspace dataspace(dims);
            
            // creade dataset
            hid_t dataset_id = H5Dcreate(group.id, name.c_str(), primitive_type_wrapper<T>::hdf5_type_id(), 
                                         dataspace.id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (dataset_id < 0)
                error(__FILE__, __LINE__, "error in H5Dcreate()");

            // write data
            if (H5Dwrite(dataset_id, primitive_type_wrapper<T>::hdf5_type_id(), dataspace.id, H5S_ALL, 
                         H5P_DEFAULT, data) < 0)
                error(__FILE__, __LINE__, "error in H5Dwrite()");

            // close dataset
            if (H5Dclose(dataset_id) < 0)
                error(__FILE__, __LINE__, "error in H5Dclose()");
        }

        template <typename T>
        void write(const std::string& name, T* data, int size = 1)
        {
            std::vector<int> dims(1);
            dims[0] = size;
            write(name, data, dims);
        }

        template <typename T, int N>
        void write(const std::string& name, mdarray<T,N>& data)
        {
            std::vector<int> dims(N);
            for (int i = 0; i < N; i++)
                dims[i] = data.size(i);

            write(name, data.get_ptr(), dims);
        }

        template<typename T>
        void read(const std::string& name, T* data, const std::vector<int>& dims)
        {
            hdf5_group group(file_id_, path_);

            hdf5_dataspace dataspace(dims);

            hdf5_dataset dataset(group.id, name);

            if (H5Dread(dataset.id, primitive_type_wrapper<T>::hdf5_type_id(), dataspace.id, H5S_ALL, 
                        H5P_DEFAULT, data) < 0)
                error(__FILE__, __LINE__, "error in H5Dread()");
        }

        template <typename T, int N>
        void read(const std::string& name, mdarray<T,N>& data)
        {
            std::vector<int> dims(N);
            for (int i = 0; i < N; i++)
                dims[i] = data.size(i);

            read(name, data.get_ptr(), dims);

        }

        hdf5_tree operator[](const std::string& path__)
        {
            std::string new_path = path_ + "/" + path__;
            return hdf5_tree(file_id_, new_path);
        }
};

};

