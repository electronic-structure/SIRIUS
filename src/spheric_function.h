#ifndef __SPHERIC_FUNCTION_H__
#define __SPHERIC_FUNCTION_H__

#include <typeinfo>
#include "radial_grid.h"
#include "spline.h"

namespace sirius
{

template <typename T> 
class Spheric_function
{
    private:

        /// function data
        mdarray<T, 2> data_;

        Radial_grid& radial_grid_;

        SHT* sht_;

        int angular_domain_size_;
        int angular_domain_idx_;

        int radial_domain_size_;
        int radial_domain_idx_;

    public:

        template <typename U> 
        friend class Spheric_function;

        Spheric_function(Radial_grid& radial_grid__, int angular_domain_size__) 
            : radial_grid_(radial_grid__), 
              sht_(NULL), 
              angular_domain_size_(angular_domain_size__), 
              angular_domain_idx_(1),
              radial_domain_size_(radial_grid__.num_mt_points()), 
              radial_domain_idx_(0)
        {
            data_.set_dimensions(radial_domain_size_, angular_domain_size_);
            data_.allocate();
        }
        
        Spheric_function(int angular_domain_size__, Radial_grid& radial_grid__) 
            : radial_grid_(radial_grid__), 
              sht_(NULL), 
              angular_domain_size_(angular_domain_size__), 
              angular_domain_idx_(0),
              radial_domain_size_(radial_grid__.num_mt_points()), 
              radial_domain_idx_(1)
        {
            data_.set_dimensions(angular_domain_size_, radial_domain_size_);
            data_.allocate();
        }

        Spheric_function(T* ptr, int angular_domain_size__, Radial_grid& radial_grid__) 
            : radial_grid_(radial_grid__), 
              sht_(NULL), 
              angular_domain_size_(angular_domain_size__), 
              angular_domain_idx_(0),
              radial_domain_size_(radial_grid__.num_mt_points()), 
              radial_domain_idx_(1)
        {
            data_.set_dimensions(angular_domain_size_, radial_domain_size_);
            data_.set_ptr(ptr);
        }

        Spheric_function(SHT& sht__, Radial_grid& radial_grid__) 
            : radial_grid_(radial_grid__), 
              sht_(&sht__), 
              angular_domain_size_(sht__.num_points()), 
              angular_domain_idx_(0),
              radial_domain_size_(radial_grid__.num_mt_points()), 
              radial_domain_idx_(1)
        {
            data_.set_dimensions(angular_domain_size_, radial_domain_size_);
            data_.allocate();
        }

        Spheric_function(T* ptr, SHT& sht__, Radial_grid& radial_grid__) 
            : radial_grid_(radial_grid__), 
              sht_(&sht__), 
              angular_domain_size_(sht__.num_points()), 
              angular_domain_idx_(0),
              radial_domain_size_(radial_grid__.num_mt_points()), 
              radial_domain_idx_(1)
        {
            data_.set_dimensions(angular_domain_size_, radial_domain_size_);
            data_.set_ptr(ptr);
        }
        
        template <typename U>
        Spheric_function(Spheric_function<U>& f, bool fill) 
            : radial_grid_(f.radial_grid_), 
              sht_(f.sht_), 
              angular_domain_size_(f.angular_domain_size_),
              angular_domain_idx_(f.angular_domain_idx_), 
              radial_domain_size_(f.radial_domain_size_),
              radial_domain_idx_(f.radial_domain_idx_)
        {
            if (radial_domain_idx_ == 0)
            {
                data_.set_dimensions(radial_domain_size_, angular_domain_size_);
            }
            else
            {
                data_.set_dimensions(angular_domain_size_, radial_domain_size_);
            }
            data_.allocate();

            if (fill)
            {
                if (typeid(T) != typeid(U))
                {
                    f.sh_convert(*this);
                }
                else
                {
                    memcpy(this->data_.ptr(), &f(0, 0), this->data_.size() * sizeof(T));
                }
            }
        }

        template <typename U>
        void sh_convert(Spheric_function<U>& f);
        
        void sh_transform(Spheric_function<T>& f);

        inline int angular_domain_size()
        {
            return angular_domain_size_;
        }

        inline int angular_domain_idx()
        {
            return angular_domain_idx_;
        }

        inline int radial_domain_size()
        {
            return radial_domain_size_;
        }

        inline int radial_domain_idx()
        {
            return radial_domain_idx_;
        }

        inline Radial_grid& radial_grid()
        {
            return radial_grid_;
        }

        inline T& operator()(const int64_t i0, const int64_t i1) 
        {
            return data_(i0, i1);
        }

        void zero()
        {
            data_.zero();
        }

        void allocate()
        {
            data_.allocate();
        }

        void set_ptr(T* ptr)
        {
            data_.set_ptr(ptr);
        }

        void add(Spheric_function<T>& f)
        {
            for (int64_t i1 = 0; i1 < (int64_t)data_.size(1); i1++)
            {
                for (int64_t i0 = 0; i0 < (int64_t)data_.size(0); i0++) data_(i0, i1) += f(i0, i1);
            }
        }
        
        void copy(Spheric_function<T>& f)
        {
            for (int64_t i1 = 0; i1 < (int64_t)data_.size(1); i1++)
            {
                for (int64_t i0 = 0; i0 < (int64_t)data_.size(0); i0++) data_(i0, i1) = f(i0, i1);
            }
        }
};

#include "spheric_function.hpp"

}

#endif // __SPHERIC_FUNCTION_H__
