// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file serializer.hpp
 *
 *  \brief Serializer for simple data structures.
 */

#ifndef __SERIALIZER_HPP__
#define __SERIALIZER_HPP__

#include "communicator.hpp"

namespace sddk {

struct serializer
{
    /// Position in the stream. This is used during unpacking.
    size_t pos{0};
    /// Data stream is represendted as a sequence of characters.
    std::vector<uint8_t> stream;

    uint8_t* expand(size_t nbytes__)
    {
        stream.resize(stream.size() + nbytes__);
        return &stream[stream.size() - nbytes__];
    }

    void send_recv(Communicator const& comm__, int source__, int dest__)
    {
        if (source__ == dest__) {
            return;
        }

        size_t sz;

        Request r1, r2;

        int tag = Communicator::get_tag(source__, dest__);

        if (comm__.rank() == source__) {
            sz = stream.size();
            assert(sz < std::numeric_limits<int>::max());
            r1 = comm__.isend(&sz, 1, dest__, tag++);
            r2 = comm__.isend(&stream[0], (int)sz, dest__, tag++);
        }

        if (comm__.rank() == dest__) {
            comm__.recv(&sz, 1, source__, tag++);
            stream.resize(sz);
            comm__.recv(&stream[0], (int)sz, source__, tag++);
        }

        if (comm__.rank() == source__) {
            r1.wait();
            r2.wait();
        }
    }
};

template <typename T>
inline void serialize(serializer& s__, T var__)
{
    size_t sz = sizeof(T);
    auto end = s__.expand(sz);
    std::memcpy(end, reinterpret_cast<uint8_t*>(&var__), sz);
}

template <typename T>
inline void deserialize(serializer& s__, T& var__)
{
    size_t sz = sizeof(T);
    std::memcpy(reinterpret_cast<uint8_t*>(&var__), &s__.stream[s__.pos], sz);
    s__.pos += sz;
}

template <typename T>
inline void serialize(serializer& s__, std::vector<T> const& vec__)
{
    serialize(s__, vec__.size());
    size_t sz = sizeof(T) * vec__.size();
    auto end = s__.expand(sz);
    std::memcpy(end, reinterpret_cast<uint8_t const*>(&vec__[0]), sz);
}

template <typename T>
inline void deserialize(serializer& s__, std::vector<T>& vec__)
{
    size_t sz;
    deserialize(s__, sz);
    vec__.resize(sz);
    sz = sizeof(T) * vec__.size();
    std::memcpy(reinterpret_cast<uint8_t*>(&vec__[0]), &s__.stream[s__.pos], sz);
    s__.pos += sz;
}

template <typename T, int N>
void serialize(serializer& s__, mdarray<T, N> const& array__)
{
    serialize(s__, array__.size());
    if (array__.size() == 0) {
        return;
    }
    for (int i = 0; i < N; i++) {
        serialize(s__, array__.dim(i).begin());
        serialize(s__, array__.dim(i).end());
    }
    size_t sz = sizeof(T) * array__.size();
    auto end = s__.expand(sz);
    std::memcpy(end, reinterpret_cast<uint8_t const*>(&array__[0]), sz);
}

template <typename T, int N>
void deserialize(serializer& s__, mdarray<T, N>& array__)
{
    size_t sz;
    deserialize(s__, sz);
    if (sz == 0) {
        array__ = mdarray<T, N>();
        return;
    }
    std::array<mdarray_index_descriptor, N> dims;
    for (int i = 0; i < N; i++) {
        mdarray_index_descriptor::index_type begin, end;
        deserialize(s__, begin);
        deserialize(s__, end);
        dims[i] = mdarray_index_descriptor(begin, end);
    }
    array__ = mdarray<T, N>(dims);
    sz = sizeof(T) * array__.size();
    std::memcpy(&array__[0], &s__.stream[s__.pos], sz);
    s__.pos += sz;
}

inline void serialize(serializer& s__, block_data_descriptor const& dd__)
{
    serialize(s__, dd__.num_ranks);
    serialize(s__, dd__.counts);
    serialize(s__, dd__.offsets);
}

inline void deserialize(serializer& s__, block_data_descriptor& dd__)
{
    deserialize(s__, dd__.num_ranks);
    deserialize(s__, dd__.counts);
    deserialize(s__, dd__.offsets);
}

}

#endif
