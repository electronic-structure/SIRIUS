/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file hdf5_tree.hpp
 *
 *  \brief Contains definition and implementation of sirius::HDF5_tree class.
 */

#ifndef __HDF5_TREE_HPP__
#define __HDF5_TREE_HPP__

#include <fstream>
#include <hdf5.h>
#include <vector>
#include <string>
#include <initializer_list>
#include "core/memory.hpp"
#include "core/rte/rte.hpp"

namespace sirius {

enum class hdf5_access_t
{
    truncate,
    read_write,
    read_only
};

template <typename T>
struct hdf5_type_wrapper;

template <>
struct hdf5_type_wrapper<float>
{
    operator hid_t() const noexcept
    {
        return H5T_NATIVE_FLOAT;
    };
};

template <>
struct hdf5_type_wrapper<double>
{
    operator hid_t() const noexcept
    {
        return H5T_NATIVE_DOUBLE;
    };
};

template <>
struct hdf5_type_wrapper<int>
{
    operator hid_t() const noexcept
    {
        return H5T_NATIVE_INT;
    };
};

template <>
struct hdf5_type_wrapper<uint8_t>
{
    operator hid_t() const noexcept
    {
        return H5T_NATIVE_UCHAR;
    };
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
                RTE_THROW(s);
            }
        }

        /// Constructor which creates the new group.
        HDF5_group(HDF5_group const& g, const std::string& name)
        {
            if ((id_ = H5Gcreate(g.id(), name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) < 0) {
                std::stringstream s;
                s << "error in H5Gcreate()" << std::endl << "name : " << name;
                RTE_THROW(s);
            }
        }

        /// Destructor.
        ~HDF5_group()
        {
            if (H5Gclose(id_) < 0) {
                RTE_THROW("error in H5Gclose()");
            }
        }

        /// Return HDF5 id of the current object.
        inline hid_t
        id() const
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
                RTE_THROW("error in H5Screate_simple()");
            }
        }

        /// Destructor.
        ~HDF5_dataspace()
        {
            if (H5Sclose(id_) < 0) {
                RTE_THROW("error in H5Sclose()");
            }
        }

        /// Return HDF5 id of the current object.
        inline hid_t
        id() const
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
                RTE_THROW("error in H5Dopen()");
            }
        }

        /// Constructor which creates the new dataset object.
        HDF5_dataset(HDF5_group& group, HDF5_dataspace& dataspace, const std::string& name, hid_t type_id)
        {
            if ((id_ = H5Dcreate(group.id(), name.c_str(), type_id, dataspace.id(), H5P_DEFAULT, H5P_DEFAULT,
                                 H5P_DEFAULT)) < 0) {
                RTE_THROW("error in H5Dcreate()");
            }
        }

        /// Destructor.
        ~HDF5_dataset()
        {
            if (H5Dclose(id_) < 0) {
                RTE_THROW("error in H5Dclose()");
            }
        }

        /// Return HDF5 id of the current object.
        inline hid_t
        id() const
        {
            return id_;
        }
    };

    /// Auxiliary class to handle HDF5 Attribute object
    ///
    /// @remark The Attribute's Dataspace is assumed to be SCALAR.
    class HDF5_attribute
    {
      private:
        /// HDF5 id of the current object
        hid_t id_;

      public:
        /// Constructor which opens the existing attribute object
        HDF5_attribute(hid_t attribute_id, const std::string& name)
        {
            if ((id_ = H5Aopen(attribute_id, name.c_str(), H5P_DEFAULT)) < 0)
                RTE_THROW("error in H5Aopen()");
        }

        /// Constructor creating the new attribute object
        ///
        /// @remark The Attribute's DATASPACE is assumed to be SCALAR
        ///
        /// @param parent_id GROUP or DATASET ID to which the attribute is attached
        /// @param name Name of the attribute
        /// @param type_id Type ID for the attribute
        HDF5_attribute(hid_t parent_id, const std::string& name, hid_t type_id)
        {
            hid_t dataspace_id;

            // Create SCALAR Dataspace
            if ((dataspace_id = H5Screate(H5S_SCALAR)) < 0) {
                RTE_THROW("error in H5Screate()");
            }

            if ((id_ = H5Acreate(parent_id, name.c_str(), type_id, dataspace_id, H5P_DEFAULT, H5P_DEFAULT)) < 0) {
                RTE_THROW("error in H5Acreate()");
            }
        }

        /// Destructor
        ~HDF5_attribute()
        {
            if (H5Aclose(id_) < 0) {
                RTE_THROW("error in H5Aclose()");
            }
        }

        /// Return HDF5 id of the current object
        inline hid_t
        id() const
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

    /// Create HDF5 string type from std::string
    hid_t
    string_type(const std::string& data, hid_t base_string_type) const
    {
        // Create string type
        hid_t string_type = H5Tcopy(base_string_type);
        if (H5Tset_size(string_type, std::size(data)) < 0) {
            RTE_THROW("error in H5Tset_size()");
        }
        return string_type;
    }

    /// Create HDF5 array type from std::vector
    template <typename T>
    hid_t
    array_type(const std::vector<T>& data) const
    {
        hsize_t data_size = std::size(data);
        return H5Tarray_create(hdf5_type_wrapper<T>(), 1, &data_size);
    }

    /// Write a multidimensional array.
    template <typename T>
    void
    write(const std::string& name, T const* data, std::vector<int> const& dims)
    {
        /* open group */
        HDF5_group group(file_id_, path_);

        /* make dataspace */
        HDF5_dataspace dataspace(dims);

        /* create new dataset */
        HDF5_dataset dataset(group, dataspace, name, hdf5_type_wrapper<T>());

        /* write data */
        if (H5Dwrite(dataset.id(), hdf5_type_wrapper<T>(), dataspace.id(), H5S_ALL, H5P_DEFAULT, data) < 0) {
            RTE_THROW("error in H5Dwrite()");
        }
    }

    /// Write attribure to current GROUP
    ///
    /// @param name Name of the attribute
    /// @param data Attribute data
    /// @param type_id Attribute type ID
    template <typename T>
    void
    write_attribute(const std::string& name, const T* const data, hid_t type_id) const
    {
        HDF5_group group(file_id_, path_);

        HDF5_attribute attribute(group.id(), name, type_id);

        if (H5Awrite(attribute.id(), type_id, data) < 0) {
            RTE_THROW("error in H5Awrite()");
        }
    }

    /// Write attribure to DATASET within the GROUP
    ///
    /// @param name Name of the attribute
    /// @param data Attribute data
    /// @param type_id Attribute type ID
    /// @param dataset_name Dataset in current group to wich attach the attribute
    template <typename T>
    void
    write_attribute(const std::string& name, const T* const data, hid_t type_id, const std::string& dataset_name) const
    {
        HDF5_group group(file_id_, path_);

        HDF5_dataset dataset(group.id(), dataset_name);

        HDF5_attribute attribute(dataset.id(), name, type_id);

        if (H5Awrite(attribute.id(), type_id, data) < 0) {
            RTE_THROW("error in H5Awrite()");
        }
    }

    /// Read a multidimensional array.
    template <typename T>
    void
    read(const std::string& name, T* data, std::vector<int> const& dims)
    {
        HDF5_group group(file_id_, path_);

        HDF5_dataspace dataspace(dims);

        HDF5_dataset dataset(group.id(), name);

        if (H5Dread(dataset.id(), hdf5_type_wrapper<T>(), dataspace.id(), H5S_ALL, H5P_DEFAULT, data) < 0) {
            RTE_THROW("error in H5Dread()");
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
            RTE_THROW("error in H5open()");
        }

        if (false) {
            H5Eset_auto(H5E_DEFAULT, NULL, NULL);
        }

        switch (access__) {
            case hdf5_access_t::truncate: {
                file_id_ = H5Fcreate(file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                if (file_id_ < 0) {
                    RTE_THROW("error in H5Fcreate()");
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
            RTE_THROW("H5Fopen() failed");
        }

        path_ = "/";
    }

    /// Destructor.
    ~HDF5_tree()
    {
        if (root_node_) {
            if (H5Fclose(file_id_) < 0) {
                RTE_THROW("error in H5Fclose()");
            }
        }
    }

    /// Create node by integer index.
    /** Create node at the current location using integer index as a name. */
    HDF5_tree
    create_node(int idx)
    {
        return create_node(std::to_string(idx));
    }

    /// Create node by name.
    /** Create node with the given name at the current location.*/
    HDF5_tree
    create_node(std::string const& name)
    {
        /* try to open a group */
        HDF5_group group(file_id_, path_);
        /* try to create a new group */
        HDF5_group(group, name);

        return (*this)[name];
    }

    template <typename T, int N>
    void
    write(const std::string& name, mdarray<std::complex<T>, N> const& data)
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
    void
    write(std::string const& name__, mdarray<T, N> const& data__)
    {
        std::vector<int> dims(N);
        for (int i = 0; i < N; i++) {
            dims[i] = static_cast<int>(data__.size(i));
        }
        write(name__, data__.at(memory_t::host), dims);
    }

    /// Write a multidimensional array by integer index.
    template <typename T, int N>
    void
    write(int name_id, mdarray<T, N> const& data)
    {
        std::string name = std::to_string(name_id);
        write(name, data);
    }

    /// Write a buffer.
    template <typename T>
    void
    write(const std::string& name, T const* data, int size)
    {
        std::vector<int> dims(1);
        dims[0] = size;
        write(name, data, dims);
    }

    /// Write a scalar.
    template <typename T>
    void
    write(const std::string& name, T data)
    {
        std::vector<int> dims(1);
        dims[0] = 1;
        write(name, &data, dims);
    }

    /// Write a vector by id.
    template <typename T>
    void
    write(int name_id, std::vector<T> const& vec)
    {
        std::string name = std::to_string(name_id);
        write(name, &vec[0], (int)vec.size());
    }

    /// Write a vector by name.
    template <typename T>
    void
    write(const std::string& name, std::vector<T> const& vec)
    {
        write(name, &vec[0], (int)vec.size());
    }

    /// Write attribute
    template <typename T>
    void
    write_attribute(const std::string& name, const T& data) const
    {
        write_attribute(name, &data, hdf5_type_wrapper<T>());
    }

    /// Write attribute
    template <typename T>
    void
    write_attribute(const std::string& name, const T& data, const std::string& dataset_name) const
    {
        write_attribute(name, &data, hdf5_type_wrapper<T>(), dataset_name);
    }

    /// Write string attribute
    void
    write_attribute(const std::string& name, const std::string& data, hid_t base_string_type = H5T_FORTRAN_S1) const
    {
        write_attribute(name, data.c_str(), string_type(data, base_string_type));
    }

    /// Write string attribute
    void
    write_attribute(const std::string& name, const std::string& data, const std::string& dataset_name,
                    hid_t base_string_type = H5T_FORTRAN_S1) const
    {
        write_attribute(name, data.c_str(), string_type(data, base_string_type), dataset_name);
    }

    /// Write string attribute
    template <std::size_t N>
    void
    write_attribute(const std::string& name, const char (&data)[N], hid_t base_string_type = H5T_FORTRAN_S1) const
    {
        write_attribute(name, std::string(data), base_string_type);
    }

    /// Write string attribute
    template <std::size_t N>
    void
    write_attribute(const std::string& name, const char (&data)[N], const std::string& dataset_name,
                    hid_t base_string_type = H5T_FORTRAN_S1) const
    {
        write_attribute(name, std::string(data), dataset_name, base_string_type);
    }

    /// Write array attribute
    template <typename T>
    void
    write_attribute(const std::string& name, const std::vector<T>& data) const
    {
        write_attribute(name, data.data(), array_type(data));
    }

    /// Write array attribute
    template <typename T>
    void
    write_attribute(const std::string& name, const std::vector<T>& data, const std::string& dataset_name) const
    {
        write_attribute(name, data.data(), array_type(data), dataset_name);
    }

    /// Write array attribute
    template <typename T>
    void
    write_attribute(const std::string& name, const std::initializer_list<T>& data) const
    {
        write_attribute(name, std::vector<T>(data));
    }

    /// Write array attribute
    template <typename T>
    void
    write_attribute(const std::string& name, const std::initializer_list<T>& data,
                    const std::string& dataset_name) const
    {
        write_attribute(name, std::vector<T>(data), dataset_name);
    }

    template <int N>
    void
    read(const std::string& name, mdarray<std::complex<double>, N>& data)
    {
        std::vector<int> dims(N + 1);
        dims[0] = 2;
        for (int i = 0; i < N; i++) {
            dims[i + 1] = (int)data.size(i);
        }
        read(name, (double*)data.at(memory_t::host), dims);
    }

    template <typename T, int N>
    void
    read(const std::string& name, mdarray<T, N>& data)
    {
        std::vector<int> dims(N);
        for (int i = 0; i < N; i++) {
            dims[i] = (int)data.size(i);
        }
        read(name, data.at(memory_t::host), dims);
    }

    template <typename T, int N>
    void
    read(int name_id, mdarray<T, N>& data)
    {
        std::string name = std::to_string(name_id);
        read(name, data);
    }

    /// Read a vector or a scalar.
    template <typename T>
    void
    read(std::string const& name, T* data, int size)
    {
        std::vector<int> dims(1);
        dims[0] = size;
        read(name, data, dims);
    }

    template <typename T>
    void
    read(int name_id, std::vector<T>& vec)
    {
        read(std::to_string(name_id), &vec[0], (int)vec.size());
    }

    template <typename T>
    void
    read(std::string const& name, std::vector<T>& vec)
    {
        read(name, &vec[0], (int)vec.size());
    }

    HDF5_tree
    operator[](std::string const& path__)
    {
        auto new_path = path_ + path__ + "/";
        return HDF5_tree(file_id_, new_path);
    }

    HDF5_tree
    operator[](int idx__)
    {
        auto new_path = path_ + std::to_string(idx__) + "/";
        return HDF5_tree(file_id_, new_path);
    }
};

}; // namespace sirius

#endif // __HDF5_TREE_HPP__
