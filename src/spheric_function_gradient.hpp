template <typename T>
void Spheric_function_gradient<T>::gradient(Spheric_function<double_complex>& f)
{
    for (int i = 0; i < 3; i++) grad_[i]->zero();

    int lmax = Utils::lmax_by_lmmax(f.angular_domain_size());

    Spline<double_complex> s(f.radial_domain_size(), f.radial_grid());

    for (int l = 0; l <= lmax; l++)
    {
        double d1 = sqrt(double(l + 1) / double(2 * l + 3));
        double d2 = sqrt(double(l) / double(2 * l - 1));

        for (int m = -l; m <= l; m++)
        {
            int lm = Utils::lm_by_l_m(l, m);
            if (f.radial_domain_idx() == 0)
            {
                for (int ir = 0; ir < f.radial_domain_size(); ir++) s[ir] = f(ir, lm);
            }
            else
            {
                for (int ir = 0; ir < f.radial_domain_size(); ir++) s[ir] = f(lm, ir);
            }
            s.interpolate();

            for (int mu = -1; mu <= 1; mu++)
            {
                int j = (mu + 2) % 3; // map -1,0,1 to 1,2,0

                if ((l + 1) <= lmax && abs(m + mu) <= l + 1)
                {
                    int lm1 = Utils::lm_by_l_m(l + 1, m + mu); 
                    double d = d1 * SHT::clebsch_gordan(l, 1, l + 1, m, mu, m + mu);
                    if (f.radial_domain_idx() == 0)
                    {
                        for (int ir = 0; ir < f.radial_domain_size(); ir++)
                            (*grad_[j])(ir, lm1) += (s.deriv(1, ir) - f(ir, lm) * f.radial_grid().x_inv(ir) * double(l)) * d;  
                    }
                    else
                    {
                        for (int ir = 0; ir < f.radial_domain_size(); ir++)
                            (*grad_[j])(lm1, ir) += (s.deriv(1, ir) - f(lm, ir) * f.radial_grid().x_inv(ir) * double(l)) * d;  
                    }
                }
                if ((l - 1) >= 0 && abs(m + mu) <= l - 1)
                {
                    int lm1 = Utils::lm_by_l_m(l - 1, m + mu); 
                    double d = d2 * SHT::clebsch_gordan(l, 1, l - 1, m, mu, m + mu); 
                    if (f.radial_domain_idx() == 0)
                    {
                        for (int ir = 0; ir < f.radial_domain_size(); ir++)
                            (*grad_[j])(ir, lm1) -= (s.deriv(1, ir) + f(ir, lm) * f.radial_grid().x_inv(ir) * double(l + 1)) * d;
                    }
                    else
                    {
                        for (int ir = 0; ir < f.radial_domain_size(); ir++)
                            (*grad_[j])(lm1, ir) -= (s.deriv(1, ir) + f(lm, ir) * f.radial_grid().x_inv(ir) * double(l + 1)) * d;
                    }
                }
            }
        }
    }

    double_complex d1(1.0 / sqrt(2.0), 0);
    double_complex d2(0, 1.0 / sqrt(2.0));

    if (f.radial_domain_idx() == 0)
    {
        for (int lm = 0; lm < f.angular_domain_size(); lm++)
        {
            for (int ir = 0; ir < f.radial_domain_size(); ir++)
            {
                double_complex g_p = (*grad_[0])(ir, lm);
                double_complex g_m = (*grad_[1])(ir, lm);
                (*grad_[0])(ir, lm) = d1 * (g_m - g_p);
                (*grad_[1])(ir, lm) = d2 * (g_m + g_p);
            }
        }
    }
    else
    {
        for (int ir = 0; ir < f.radial_domain_size(); ir++)
        {
            for (int lm = 0; lm < f.angular_domain_size(); lm++)
            {
                double_complex g_p = (*grad_[0])(lm, ir);
                double_complex g_m = (*grad_[1])(lm, ir);
                (*grad_[0])(lm, ir) = d1 * (g_m - g_p);
                (*grad_[1])(lm, ir) = d2 * (g_m + g_p);
            }
        }
    }
}

template <typename T>
void Spheric_function_gradient<T>::gradient(Spheric_function<double>& f)
{
    Spheric_function<double_complex> zf(f, true);
    Spheric_function_gradient<double_complex> zg(zf);

    for (int i = 0; i < 3; i++) zg[i].sh_convert(*grad_[i]);
}

template<typename T>
vector3d<T> inner(Spheric_function<T>& f, Spheric_function_gradient<T>& grad)
{
    return vector3d<T>(inner(f, grad[0]), inner(f, grad[1]), inner(f, grad[2]));
}

