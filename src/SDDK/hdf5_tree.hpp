// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file hdf5_tree.hpp
 *
 *  \brief Contains definition and implementation of sddk::HDF5_tree class.
 */

#ifndef __HDF5_TREE_HPP__
#define __HDF5_TREE_HPP__

#include <fstream>
#include <hdf5.h>
#include "memory.hpp"

namespace sddk {

enum class hdf5_access_t {
    truncate,
    read_write,
    read_only
};

template <typename T>
struct hdf5_type_wrapper;

template<>
struct hdf5_type_wrapper<double>
{
    static hid_t type_id()
    {
        return H5T_NATIVE_DOUBLE;
    }
};

template<>
struct hdf5_type_wrapper<std::complex<double>>
{
    static hid_t type_id()
    {
        return H5T_NATIVE_LDOUBLE;
    }
};

template<>
struct hdf5_type_wrapper<int>
{
    static hid_t type_id()
    {
        return H5T_NATIVE_INT;
    }
};

template<>
struct hdf5_type_wrapper<uint8_t>
{
    static hid_t type_id()
    {
        return H5T_NATIVE_UCHAR;
    }
};

/// Interface to the HDF5 library.
class HDF5_tree
{
  private:
    /// HDF5 file name
    std::string file_name_;

    /// path inside HDF5 file
    std::string path_;

    /// HDF5 file handler
    hid_t file_id_{-1};

    /// True if this is a root node
    bool root_node_{true};

    /// Auxiliary class to handle HDF5 Group object
    class HDF5_group
    {
      private:
        /// HDF5 id of the current object
        hid_t id_;

      public:
        /// Constructor which openes the existing group.
        HDF5_group(hid_t file_id, const std::string& path)
        {
            if ((id_ = H5Gopen(file_id, path.c_str(), H5P_DEFAULT)) < 0) {
                std::stringstream s;
                s << "error in H5Gopen()" << std::endl << "path : " << path;
                TERMINATE(s);
            }
        }

        /// Constructor which creates the new group.
        HDF5_group(HDF5_group const& g, const std::string& name)
        {
            if ((id_ = H5Gcreate(g.id(), name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) < 0) {
                std::stringstream s;
                s << "error in H5Gcreate()" << std::endl << "name : " << name;
                TERMINATE(s);
            }
        }

        /// Destructor.
        ~HDF5_group()
        {
            if (H5Gclose(id_) < 0) {
                TERMINATE("error in H5Gclose()");
            }
        }

        /// Return HDF5 id of the current object.
        inline hid_t id() const
        {
            return id_;
        }
    };

    /// Auxiliary class to handle HDF5 Dataspace object
    class HDF5_dataspace
    {
      private:
        /// HDF5 id of the current object
        hid_t id_;

      public:
        /// Constructor which creates the new dataspace object.
        HDF5_dataspace(const std::vector<int> dims)
        {
            std::vector<hsize_t> current_dims(dims.size());
            for (int i = 0; i < (int)dims.size(); i++) {
                current_dims[dims.size() - i - 1] = dims[i];
            }

            if ((id_ = H5Screate_simple((int)dims.size(), &current_dims[0], NULL)) < 0) {
                TERMINATE("error in H5Screate_simple()");
            }
        }

        /// Destructor.
        ~HDF5_dataspace()
        {
            if (H5Sclose(id_) < 0) {
                TERMINATE("error in H5Sclose()");
            }
        }

        /// Return HDF5 id of the current object.
        inline hid_t id() const
        {
            return id_;
        }
    };

    /// Auxiliary class to handle HDF5 Dataset object
    class HDF5_dataset
    {
      private:
        /// HDF5 id of the current object
        hid_t id_;

      public:
        /// Constructor which openes the existing dataset object.
        HDF5_dataset(hid_t group_id, const std::string& name)
        {
            if ((id_ = H5Dopen(group_id, name.c_str(), H5P_DEFAULT)) < 0) {
                TERMINATE("error in H5Dopen()");
            }
        }

        /// Constructor which creates the new dataset object.
        HDF5_dataset(HDF5_group& group, HDF5_dataspace& dataspace, const std::string& name, hid_t type_id)
        {
            if ((id_ = H5Dcreate(group.id(), name.c_str(), type_id, dataspace.id(), H5P_DEFAULT, H5P_DEFAULT,
                                 H5P_DEFAULT)) < 0) {
                TERMINATE("error in H5Dcreate()");
            }
        }

        /// Destructor.
        ~HDF5_dataset()
        {
            if (H5Dclose(id_) < 0) {
                TERMINATE("error in H5Dclose()");
            }
        }

        /// Return HDF5 id of the current object.
        inline hid_t id() const
        {
            return id_;
        }
    };

    /// Constructor to create branches of the HDF5 tree.
    HDF5_tree(hid_t file_id__, const std::string& path__)
        : path_(path__)
        , file_id_(file_id__)
        , root_node_(false)
    {
    }

    /// Write a multidimensional array.
    template <typename T>
    void write(const std::string& name, T const* data, std::vector<int> const& dims)
    {
        /* open group */
        HDF5_group group(file_id_, path_);

        /* make dataspace */
        HDF5_dataspace dataspace(dims);

        /* create new dataset */
        HDF5_dataset dataset(group, dataspace, name, hdf5_type_wrapper<T>::type_id());

        /* write data */
        if (H5Dwrite(dataset.id(), hdf5_type_wrapper<T>::type_id(), dataspace.id(), H5S_ALL, H5P_DEFAULT, data) < 0) {
            TERMINATE("error in H5Dwrite()");
        }
    }

    /// Read a multidimensional array.
    template <typename T>
    void read(const std::string& name, T* data, std::vector<int> const& dims)
    {
        HDF5_group group(file_id_, path_);

        HDF5_dataspace dataspace(dims);

        HDF5_dataset dataset(group.id(), name);

        if (H5Dread(dataset.id(), hdf5_type_wrapper<T>::type_id(), dataspace.id(), H5S_ALL, H5P_DEFAULT, data) < 0) {
            TERMINATE("error in H5Dread()");
        }
    }

    // HDF5_tree(HDF5_tree const& src) = delete;

    // HDF5_tree& operator=(HDF5_tree const& src) = delete;

  public:

    /// Constructor to create the HDF5 tree.
    HDF5_tree(const std::string& file_name__, hdf5_access_t access__)
        : file_name_(file_name__)
    {
        if (H5open() < 0) {
            TERMINATE("error in H5open()");
        }

        if (false) {
            H5Eset_auto(H5E_DEFAULT, NULL, NULL);
        }

        switch (access__) {
            case hdf5_access_t::truncate: {
                file_id_ = H5Fcreate(file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                if (file_id_ < 0) {
                    TERMINATE("error in H5Fcreate()");
                }
                break;
            }
            case hdf5_access_t::read_write: {
                file_id_ = H5Fopen(file_name_.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                break;
            }
            case hdf5_access_t::read_only: {
                file_id_ = H5Fopen(file_name_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
                break;
            }
        }
        if (file_id_ < 0) {
            TERMINATE("H5Fopen() failed");
        }

        path_ = "/";
    }

    /// Destructor.
    ~HDF5_tree()
    {
        if (root_node_) {
            if (H5Fclose(file_id_) < 0) {
                TERMINATE("error in H5Fclose()");
            }
        }
    }

    /// Create node by integer index.
    /** Create node at the current location using integer index as a name. */
    HDF5_tree create_node(int idx)
    {
        std::stringstream s;
        s << idx;
        return create_node(s.str());
    }

    /// Create node by name.
    /** Create node with the given name at the current location.*/
    HDF5_tree create_node(std::string const& name)
    {
        /* try to open a group */
        HDF5_group group(file_id_, path_);
        /* try to create a new group */
        HDF5_group(group, name);

        return (*this)[name];
    }

    template <typename T, int N>
    void write(const std::string& name, mdarray<std::complex<T>, N> const& data)
    {
        std::vector<int> dims(N + 1);
        dims[0] = 2;
        for (int i = 0; i < N; i++) {
            dims[i + 1] = (int)data.size(i);
        }
        write(name, (T*)data.at(memory_t::host), dims);
    }

    /// Write a multidimensional array by name.
    template <typename T, int N>
    void write(std::string const& name__, mdarray<T, N> const& data__)
    {
        std::vector<int> dims(N);
        for (int i = 0; i < N; i++) {
            dims[i] = static_cast<int>(data__.size(i));
        }
        write(name__, data__.at(memory_t::host), dims);
    }

    /// Write a multidimensional array by integer index.
    template <typename T, int N>
    void write(int name_id, mdarray<T, N> const& data)
    {
        std::string name = std::to_string(name_id);
        write(name, data);
    }

    /// Write a buffer.
    template <typename T>
    void write(const std::string& name, T const* data, int size)
    {
        std::vector<int> dims(1);
        dims[0] = size;
        write(name, data, dims);
    }

    /// Write a scalar.
    template <typename T>
    void write(const std::string& name, T data)
    {
        std::vector<int> dims(1);
        dims[0] = 1;
        write(name, &data, dims);
    }

    /// Write a vector by id.
    template <typename T>
    void write(int name_id, std::vector<T> const& vec)
    {
        std::string name = std::to_string(name_id);
        write(name, &vec[0], (int)vec.size());
    }

    /// Write a vector by name.
    template <typename T>
    void write(const std::string& name, std::vector<T> const& vec)
    {
        write(name, &vec[0], (int)vec.size());
    }

    template <int N>
    void read(const std::string& name, mdarray<std::complex<double>, N>& data)
    {
        std::vector<int> dims(N + 1);
        dims[0] = 2;
        for (int i = 0; i < N; i++) {
            dims[i + 1] = (int)data.size(i);
        }
        read(name, (double*)data.at(memory_t::host), dims);
    }

    template <typename T, int N>
    void read(const std::string& name, mdarray<T, N>& data)
    {
        std::vector<int> dims(N);
        for (int i = 0; i < N; i++) {
            dims[i] = (int)data.size(i);
        }
        read(name, data.at(memory_t::host), dims);
    }

    template <typename T, int N>
    void read(int name_id, mdarray<T, N>& data)
    {
        std::string name = std::to_string(name_id);
        read(name, data);
    }

    /// Read a vector or a scalar.
    template <typename T>
    void read(const std::string& name, T* data, int size)
    {
        std::vector<int> dims(1);
        dims[0] = size;
        read(name, data, dims);
    }

    template <typename T>
    void read(int name_id, std::vector<T>& vec)
    {
        std::string name = std::to_string(name_id);
        read(name, &vec[0], (int)vec.size());
    }

    template <typename T>
    void read(const std::string& name, std::vector<T>& vec)
    {
        read(name, &vec[0], (int)vec.size());
    }

    HDF5_tree operator[](const std::string& path__)
    {
        std::string new_path = path_ + path__ + "/";
        return HDF5_tree(file_id_, new_path);
    }

    HDF5_tree operator[](int idx)
    {
        std::stringstream s;
        s << idx;
        std::string new_path = path_ + s.str() + "/";
        return HDF5_tree(file_id_, new_path);
    }
};

}; // namespace sddk

#endif // __HDF5_TREE_HPP__
