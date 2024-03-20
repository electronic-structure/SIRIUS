/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file serializer.hpp
 *
 *  \brief Serializer for simple data structures.
 */

#ifndef __SERIALIZER_HPP__
#define __SERIALIZER_HPP__

#include <limits>
#include "core/mpi/communicator.hpp"
#include "core/memory.hpp"

namespace sirius {

/// Serialize and deserialize objects.
class serializer
{
  private:
    /// Position in the stream. This is used during unpacking.
    size_t pos_{0};
    /// Data stream is represendted as a sequence of characters.
    std::vector<uint8_t> stream_;

  public:
    /// Copy n bytes into a serialization stream.
    void
    copyin(uint8_t const* ptr__, size_t nbytes__)
    {
        /* resize the array */
        stream_.resize(stream_.size() + nbytes__);
        /* copy from pointer */
        std::memcpy(&stream_[stream_.size() - nbytes__], ptr__, nbytes__);
    }

    /// Copy n bytes from the serialization stream.
    /** When data is copied out, the position inside a stream is shifted to n bytes forward. */
    void
    copyout(uint8_t* ptr__, size_t nbytes__)
    {
        std::memcpy(ptr__, &stream_[pos_], nbytes__);
        pos_ += nbytes__;
    }

    void
    send_recv(mpi::Communicator const& comm__, int source__, int dest__)
    {
        if (source__ == dest__) {
            return;
        }

        size_t sz;

        mpi::Request r1, r2;

        int tag = mpi::Communicator::get_tag(source__, dest__);

        if (comm__.rank() == source__) {
            sz = stream_.size();
            RTE_ASSERT(sz < static_cast<size_t>(std::numeric_limits<int>::max()));
            r1 = comm__.isend(&sz, 1, dest__, tag++);
            r2 = comm__.isend(&stream_[0], (int)sz, dest__, tag++);
        }

        if (comm__.rank() == dest__) {
            comm__.recv(&sz, 1, source__, tag++);
            stream_.resize(sz);
            comm__.recv(&stream_[0], (int)sz, source__, tag++);
        }

        if (comm__.rank() == source__) {
            r1.wait();
            r2.wait();
        }
    }

    std::vector<uint8_t> const&
    stream() const
    {
        return stream_;
    }
};

/// Serialize a single element.
template <typename T>
inline void
serialize(serializer& s__, T var__)
{
    s__.copyin(reinterpret_cast<uint8_t const*>(&var__), sizeof(T));
}

/// Deserialize a single element.
template <typename T>
inline void
deserialize(serializer& s__, T& var__)
{
    s__.copyout(reinterpret_cast<uint8_t*>(&var__), sizeof(T));
}

/// Serialize a vector.
template <typename T>
inline void
serialize(serializer& s__, std::vector<T> const& vec__)
{
    serialize(s__, vec__.size());
    s__.copyin(reinterpret_cast<uint8_t const*>(&vec__[0]), sizeof(T) * vec__.size());
}

/// Deserialize a vector.
template <typename T>
inline void
deserialize(serializer& s__, std::vector<T>& vec__)
{
    size_t sz;
    deserialize(s__, sz);
    vec__.resize(sz);
    s__.copyout(reinterpret_cast<uint8_t*>(&vec__[0]), sizeof(T) * vec__.size());
}

/// Serialize multidimentional array.
template <typename T, int N>
void
serialize(serializer& s__, mdarray<T, N> const& array__)
{
    serialize(s__, array__.size());
    if (array__.size() == 0) {
        return;
    }
    for (int i = 0; i < N; i++) {
        serialize(s__, array__.dim(i).begin());
        serialize(s__, array__.dim(i).end());
    }
    s__.copyin(reinterpret_cast<uint8_t const*>(&array__[0]), sizeof(T) * array__.size());
}

/// Deserialize multidimentional array.
template <typename T, int N>
void
deserialize(serializer& s__, mdarray<T, N>& array__)
{
    size_t sz;
    deserialize(s__, sz);
    if (sz == 0) {
        array__ = mdarray<T, N>();
        return;
    }
    std::array<index_range, N> dims;
    for (int i = 0; i < N; i++) {
        index_range::index_type begin, end;
        deserialize(s__, begin);
        deserialize(s__, end);
        dims[i] = index_range(begin, end);
    }
    array__ = mdarray<T, N>(dims);
    s__.copyout(reinterpret_cast<uint8_t*>(&array__[0]), sizeof(T) * array__.size());
}

/// Serialize block data descriptor.
inline void
serialize(serializer& s__, mpi::block_data_descriptor const& dd__)
{
    serialize(s__, dd__.num_ranks);
    serialize(s__, dd__.counts);
    serialize(s__, dd__.offsets);
}

/// Deserialize block data descriptor.
inline void
deserialize(serializer& s__, mpi::block_data_descriptor& dd__)
{
    deserialize(s__, dd__.num_ranks);
    deserialize(s__, dd__.counts);
    deserialize(s__, dd__.offsets);
}

} // namespace sirius

#endif
