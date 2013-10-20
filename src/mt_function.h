namespace sirius
{

//== template <typename T> 
//== class Spheric_function
//== {
//==     private:
//== 
//==         /// function data
//==         mdarray<T, 2> data_;
//== 
//==         Radial_grid& radial_grid_;
//== 
//==         SHT* sht_;
//== 
//==         int angular_domain_size_;
//==         int angular_domain_idx_;
//== 
//==         int radial_domain_size_;
//==         int radial_domain_idx_;
//== 
//==     public:
//== 
//==         template <typename U> 
//==         friend class Spheric_function;
//== 
//==         Spheric_function(Radial_grid& radial_grid__, int angular_domain_size__) : 
//==             radial_grid_(radial_grid__), sht_(NULL), angular_domain_size_(angular_domain_size__), angular_domain_idx_(1),
//==             radial_domain_size_(radial_grid__.num_mt_points()), radial_domain_idx_(0)
//==         {
//==             data_.set_dimensions(radial_domain_size_, angular_domain_size_);
//==             data_.allocate();
//==         }
//==         
//==         Spheric_function(int angular_domain_size__, Radial_grid& radial_grid__) : 
//==             radial_grid_(radial_grid__), sht_(NULL), angular_domain_size_(angular_domain_size__), angular_domain_idx_(0),
//==             radial_domain_size_(radial_grid__.num_mt_points()), radial_domain_idx_(1)
//==         {
//==             data_.set_dimensions(angular_domain_size_, radial_domain_size_);
//==             data_.allocate();
//==         }
//== 
//==         Spheric_function(T* ptr, int angular_domain_size__, Radial_grid& radial_grid__) : 
//==             radial_grid_(radial_grid__), sht_(NULL), angular_domain_size_(angular_domain_size__), angular_domain_idx_(0),
//==             radial_domain_size_(radial_grid__.num_mt_points()), radial_domain_idx_(1)
//==         {
//==             data_.set_dimensions(angular_domain_size_, radial_domain_size_);
//==             data_.set_ptr(ptr);
//==         }
//== 
//==         Spheric_function(SHT& sht__, Radial_grid& radial_grid__) : 
//==             radial_grid_(radial_grid__), sht_(&sht__), angular_domain_size_(sht__.num_points()), angular_domain_idx_(0),
//==             radial_domain_size_(radial_grid__.num_mt_points()), radial_domain_idx_(1)
//==         {
//==             data_.set_dimensions(angular_domain_size_, radial_domain_size_);
//==             data_.allocate();
//==         }
//== 
//==         Spheric_function(T* ptr, SHT& sht__, Radial_grid& radial_grid__) : 
//==             radial_grid_(radial_grid__), sht_(&sht__), angular_domain_size_(sht__.num_points()), angular_domain_idx_(0),
//==             radial_domain_size_(radial_grid__.num_mt_points()), radial_domain_idx_(1)
//==         {
//==             data_.set_dimensions(angular_domain_size_, radial_domain_size_);
//==             data_.set_ptr(ptr);
//==         }
//==         
//==         template <typename U>
//==         Spheric_function(Spheric_function<U>& f, bool fill) : 
//==             radial_grid_(f.radial_grid_), sht_(f.sht_), angular_domain_size_(f.angular_domain_size_),
//==             angular_domain_idx_(f.angular_domain_idx_), radial_domain_size_(f.radial_domain_size_),
//==             radial_domain_idx_(f.radial_domain_idx_)
//==         {
//==             if (radial_domain_idx_ == 0)
//==             {
//==                 data_.set_dimensions(radial_domain_size_, angular_domain_size_);
//==             }
//==             else
//==             {
//==                 data_.set_dimensions(angular_domain_size_, radial_domain_size_);
//==             }
//==             data_.allocate();
//== 
//==             if (fill)
//==             {
//==                 if (typeid(T) != typeid(U))
//==                 {
//==                     f.sh_convert(*this);
//==                 }
//==                 else
//==                 {
//==                     memcpy(this->data_.get_ptr(), &f(0, 0), this->data_.size() * sizeof(T));
//==                 }
//==             }
//==         }
//== 
//==         inline int angular_domain_size()
//==         {
//==             return angular_domain_size_;
//==         }
//== 
//==         inline int angular_domain_idx()
//==         {
//==             return angular_domain_idx_;
//==         }
//== 
//==         inline int radial_domain_size()
//==         {
//==             return radial_domain_size_;
//==         }
//== 
//==         inline int radial_domain_idx()
//==         {
//==             return radial_domain_idx_;
//==         }
//== 
//==         inline Radial_grid& radial_grid()
//==         {
//==             return radial_grid_;
//==         }
//== 
//==         inline T& operator()(const int i0, const int i1) 
//==         {
//==             return data_(i0, i1);
//==         }
//== 
//==         template <typename U>
//==         void sh_convert(Spheric_function<U>& f)
//==         {
//==             // check radial arguments
//==             if (radial_domain_idx_ != f.radial_domain_idx_ || &radial_grid_ != &f.radial_grid_)
//==             {
//==                 error_local(__FILE__, __LINE__, "wrong radial arguments");
//==             }
//== 
//==             // check angular arguments
//==             if (angular_domain_idx_ != f.angular_domain_idx_ || angular_domain_size_ != f.angular_domain_size_)
//==             {
//==                 error_local(__FILE__, __LINE__, "wrong angular argumens");
//==             }
//==             
//==             int lmax = Utils::lmax_by_lmmax(angular_domain_size_);
//== 
//==             // cache transformation arrays
//==             std::vector<complex16> tpp(angular_domain_size_);
//==             std::vector<complex16> tpm(angular_domain_size_);
//==             for (int l = 0; l <= lmax; l++)
//==             {
//==                 for (int m = -l; m <= l; m++) 
//==                 {
//==                     int lm = Utils::lm_by_l_m(l, m);
//==                     if (primitive_type_wrapper<T>::is_real() && primitive_type_wrapper<U>::is_complex())
//==                     {
//==                         tpp[lm] = SHT::ylm_dot_rlm(l, m, m);
//==                         tpm[lm] = SHT::ylm_dot_rlm(l, m, -m);
//==                     }
//==                     if (primitive_type_wrapper<T>::is_complex() && primitive_type_wrapper<U>::is_real())
//==                     {
//==                         tpp[lm] = SHT::rlm_dot_ylm(l, m, m);
//==                         tpm[lm] = SHT::rlm_dot_ylm(l, m, -m);
//==                     }
//==                 }
//==             }
//== 
//==             // radial index is first
//==             if (radial_domain_idx_ == 0)
//==             {
//==                 int lm = 0;
//==                 for (int l = 0; l <= lmax; l++)
//==                 {
//==                     for (int m = -l; m <= l; m++)
//==                     {
//==                         if (m == 0)
//==                         {
//==                             for (int ir = 0; ir < radial_domain_size_; ir++) f(ir, lm) = primitive_type_wrapper<U>::sift(this->data_(ir, lm));
//==                         }
//==                         else 
//==                         {
//==                             int lm1 = Utils::lm_by_l_m(l, -m);
//==                             for (int ir = 0; ir < radial_domain_size_; ir++)
//==                             {
//==                                 f(ir, lm) = primitive_type_wrapper<U>::sift(tpp[lm] * this->data_(ir, lm) + 
//==                                                                             tpm[lm] * this->data_(ir, lm1));
//==                             }
//==                         }
//==                         lm++;
//==                     }
//==                 }
//==             }
//==             else
//==             {
//==                 for (int ir = 0; ir < radial_domain_size_; ir++)
//==                 {
//==                     int lm = 0;
//==                     for (int l = 0; l <= lmax; l++)
//==                     {
//==                         for (int m = -l; m <= l; m++)
//==                         {
//==                             if (m == 0)
//==                             {
//==                                 f(lm, ir) = primitive_type_wrapper<U>::sift(this->data_(lm, ir));
//==                             }
//==                             else 
//==                             {
//==                                 int lm1 = Utils::lm_by_l_m(l, -m);
//==                                 f(lm, ir) = primitive_type_wrapper<U>::sift(tpp[lm] * this->data_(lm, ir) + 
//==                                                                             tpm[lm] * this->data_(lm1, ir));
//==                             }
//==                             lm++;
//==                         }
//==                     }
//==                 }
//==             }
//==         }
//== 
//==         void sh_transform(Spheric_function<T>& f)
//==         {
//==             // check radial arguments
//==             if (radial_domain_idx_ != f.radial_domain_idx_ || &radial_grid_ != &f.radial_grid_)
//==             {
//==                 error_local(__FILE__, __LINE__, "wrong radial arguments");
//==             }
//==             if (radial_domain_idx_ != 1)
//==             {
//==                 error_local(__FILE__, __LINE__, "radial argument must be second");
//==             }
//==             if ((sht_ == NULL && f.sht_ == NULL) || (sht_ != NULL && f.sht_ != NULL))
//==             {
//==                 error_local(__FILE__, __LINE__, "wrong anguler arguments");
//==             }
//==                 
//==             if (sht_ == NULL)
//==             {
//==                 if (data_.size(0) != f.sht_->lmmax()) error_local(__FILE__, __LINE__, "wrong lm size");
//==                 if (f.data_.size(0) != f.sht_->num_points()) error_local(__FILE__, __LINE__, "wrong tp size");
//==                 
//==                 f.sht_->backward_transform(&data_(0, 0), angular_domain_size_, radial_domain_size_, &f(0, 0));
//==             }
//==             
//==             if (sht_)
//==             {
//==                 if (data_.size(0) != sht_->num_points()) error_local(__FILE__, __LINE__, "wrong tp size");
//==                 if (f.data_.size(0) != sht_->lmmax()) error_local(__FILE__, __LINE__, "wrong lm size");
//==                 
//==                 sht_->forward_transform(&data_(0, 0), f.angular_domain_size_, radial_domain_size_, &f(0, 0));
//==             }
//==         }
//== 
//==         void zero()
//==         {
//==             data_.zero();
//==         }
//== 
//==         void allocate()
//==         {
//==             data_.allocate();
//==         }
//== 
//==         void set_ptr(T* ptr)
//==         {
//==             data_.set_ptr(ptr);
//==         }
//== 
//==         void add(Spheric_function<T>& f)
//==         {
//==             for (int i1 = 0; i1 < data_.size(1); i1++)
//==             {
//==                 for (int i0 = 0; i0 < data_.size(0); i0++) data_(i0, i1) += f(i0, i1);
//==             }
//==         }
//==         
//==         void copy(Spheric_function<T>& f)
//==         {
//==             for (int i1 = 0; i1 < data_.size(1); i1++)
//==             {
//==                 for (int i0 = 0; i0 < data_.size(0); i0++) data_(i0, i1) = f(i0, i1);
//==             }
//==         }
//== };

//== template <typename T>
//== class Spheric_function_gradient
//== {
//==     private:
//==         Spheric_function<T>* grad_[3];
//== 
//==         // forbid copy constructor
//==         Spheric_function_gradient(const Spheric_function_gradient& src);
//== 
//==         // forbid assigment operator
//==         Spheric_function_gradient& operator=(const Spheric_function_gradient& src);
//== 
//==         void gradient(Spheric_function<complex16>& f)
//==         {
//==             for (int i = 0; i < 3; i++) grad_[i]->zero();
//== 
//==             int lmax = Utils::lmax_by_lmmax(f.angular_domain_size());
//== 
//==             Spline<complex16> s(f.radial_domain_size(), f.radial_grid());
//== 
//==             for (int l = 0; l <= lmax; l++)
//==             {
//==                 double d1 = sqrt(double(l + 1) / double(2 * l + 3));
//==                 double d2 = sqrt(double(l) / double(2 * l - 1));
//== 
//==                 for (int m = -l; m <= l; m++)
//==                 {
//==                     int lm = Utils::lm_by_l_m(l, m);
//==                     if (f.radial_domain_idx() == 0)
//==                     {
//==                         for (int ir = 0; ir < f.radial_domain_size(); ir++) s[ir] = f(ir, lm);
//==                     }
//==                     else
//==                     {
//==                         for (int ir = 0; ir < f.radial_domain_size(); ir++) s[ir] = f(lm, ir);
//==                     }
//==                     s.interpolate();
//== 
//==                     for (int mu = -1; mu <= 1; mu++)
//==                     {
//==                         int j = (mu + 2) % 3; // map -1,0,1 to 1,2,0
//== 
//==                         if ((l + 1) <= lmax && abs(m + mu) <= l + 1)
//==                         {
//==                             int lm1 = Utils::lm_by_l_m(l + 1, m + mu); 
//==                             double d = d1 * SHT::clebsch_gordan(l, 1, l + 1, m, mu, m + mu);
//==                             if (f.radial_domain_idx() == 0)
//==                             {
//==                                 for (int ir = 0; ir < f.radial_domain_size(); ir++)
//==                                     (*grad_[j])(ir, lm1) += (s.deriv(1, ir) - f(ir, lm) * f.radial_grid().rinv(ir) * double(l)) * d;  
//==                             }
//==                             else
//==                             {
//==                                 for (int ir = 0; ir < f.radial_domain_size(); ir++)
//==                                     (*grad_[j])(lm1, ir) += (s.deriv(1, ir) - f(lm, ir) * f.radial_grid().rinv(ir) * double(l)) * d;  
//==                             }
//==                         }
//==                         if ((l - 1) >= 0 && abs(m + mu) <= l - 1)
//==                         {
//==                             int lm1 = Utils::lm_by_l_m(l - 1, m + mu); 
//==                             double d = d2 * SHT::clebsch_gordan(l, 1, l - 1, m, mu, m + mu); 
//==                             if (f.radial_domain_idx() == 0)
//==                             {
//==                                 for (int ir = 0; ir < f.radial_domain_size(); ir++)
//==                                     (*grad_[j])(ir, lm1) -= (s.deriv(1, ir) + f(ir, lm) * f.radial_grid().rinv(ir) * double(l + 1)) * d;
//==                             }
//==                             else
//==                             {
//==                                 for (int ir = 0; ir < f.radial_domain_size(); ir++)
//==                                     (*grad_[j])(lm1, ir) -= (s.deriv(1, ir) + f(lm, ir) * f.radial_grid().rinv(ir) * double(l + 1)) * d;
//==                             }
//==                         }
//==                     }
//==                 }
//==             }
//== 
//==             complex16 d1(1.0 / sqrt(2.0), 0);
//==             complex16 d2(0, 1.0 / sqrt(2.0));
//== 
//==             if (f.radial_domain_idx() == 0)
//==             {
//==                 for (int lm = 0; lm < f.angular_domain_size(); lm++)
//==                 {
//==                     for (int ir = 0; ir < f.radial_domain_size(); ir++)
//==                     {
//==                         complex16 g_p = (*grad_[0])(ir, lm);
//==                         complex16 g_m = (*grad_[1])(ir, lm);
//==                         (*grad_[0])(ir, lm) = d1 * (g_m - g_p);
//==                         (*grad_[1])(ir, lm) = d2 * (g_m + g_p);
//==                     }
//==                 }
//==             }
//==             else
//==             {
//==                 for (int ir = 0; ir < f.radial_domain_size(); ir++)
//==                 {
//==                     for (int lm = 0; lm < f.angular_domain_size(); lm++)
//==                     {
//==                         complex16 g_p = (*grad_[0])(lm, ir);
//==                         complex16 g_m = (*grad_[1])(lm, ir);
//==                         (*grad_[0])(lm, ir) = d1 * (g_m - g_p);
//==                         (*grad_[1])(lm, ir) = d2 * (g_m + g_p);
//==                     }
//==                 }
//==             }
//==         }
//== 
//==         void gradient(Spheric_function<double>& f)
//==         {
//==             Spheric_function<complex16> zf(f, true);
//==             Spheric_function_gradient<complex16> zg(zf);
//== 
//==             for (int i = 0; i < 3; i++) zg[i].sh_convert(*grad_[i]);
//==         }
//== 
//==     public:
//== 
//==         Spheric_function_gradient(Spheric_function<T>& f)
//==         {
//==             grad_[0] = new Spheric_function<T>(f, false);
//==             grad_[1] = new Spheric_function<T>(f, false);
//==             grad_[2] = new Spheric_function<T>(f, false);
//==             gradient(f);
//==         }
//== 
//==         Spheric_function<T>& operator[](const int idx)
//==         {
//==             return *(grad_[idx]);
//==         }
//== 
//==         ~Spheric_function_gradient()
//==         {
//==             delete grad_[0];
//==             delete grad_[1];
//==             delete grad_[2];
//==         }
//== };
//== 
//== template <typename T>
//== T inner(Spheric_function<T>& f1, Spheric_function<T>& f2)
//== {
//==     if ((f1.angular_domain_idx() != f2.angular_domain_idx()) || (f1.angular_domain_size() != f2.angular_domain_size()))
//==     {
//==         error_local(__FILE__, __LINE__, "wrong angular arguments");
//==     }
//==     if ((f1.radial_domain_idx() != f2.radial_domain_idx()) || (&f1.radial_grid() != &f2.radial_grid()))
//==     {
//==         error_local(__FILE__, __LINE__, "wrong radial arguments");
//==     }
//==     Spline<T> s(f1.radial_domain_size(), f1.radial_grid());
//== 
//==     if (f1.radial_domain_idx() == 0)
//==     {
//==         for (int lm = 0; lm < f1.angular_domain_size(); lm++)
//==         {
//==             for (int ir = 0; ir < f1.radial_domain_size(); ir++)
//==                 s[ir] += primitive_type_wrapper<T>::conjugate(f1(ir, lm)) * f2(ir, lm);
//==         }       
//==     }
//==     else
//==     {
//==         for (int ir = 0; ir < f1.radial_domain_size(); ir++)
//==         {
//==             for (int lm = 0; lm < f1.angular_domain_size(); lm++)
//==                 s[ir] += primitive_type_wrapper<T>::conjugate(f1(lm, ir)) * f2(lm, ir);
//==         }
//==     }
//==     return s.interpolate().integrate(2);
//== }
//== 
//== template<typename T>
//== vector3d<T> inner(Spheric_function<T>& f, Spheric_function_gradient<T>& grad)
//== {
//==     return vector3d<T>(inner(f, grad[0]), inner(f, grad[1]), inner(f, grad[2]));
//== }
//== 




//== template <typename T>
//== class Spheric_function_vector
//== {
//==     private:
//==         std::vector< Spheric_function<T>* > vec_;
//== 
//==     public:
//==         Spheric_function_vector(T* ptr, SHT& sht, Radial_grid& radial_grid, int nd)
//==         {
//==             vec_.resize(nd);
//==             for (int i = 0; i < nd; i++) 
//==                 vec_[i] = new Spheric_function<T>(&ptr[i * sht.num_points() * radial_grid.num_mt_points()], sht, radial_grid);
//==         }
//==         ~Spheric_function_vector()
//==         {
//==             for (int i = 0; i < (int)vec_.size(); i++) delete vec_[i];
//==         }
//== 
//==         inline Spheric_function<T>& operator[](const int idx)
//==         {
//==             return *vec_[idx];
//==         }
//== 
//== };
//==             
//==         




}
