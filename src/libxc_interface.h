namespace sirius
{

class libxc_interface
{
    private:

        xc_func_type func1_[2];
        xc_func_type func2_[2];

    public:
        
        libxc_interface()
        {
            //int xc_id[] = {XC_LDA_X, XC_LDA_C_VWN};
            int xc_id[] = {XC_LDA_X, XC_LDA_C_PW};
            for (int i = 0; i < 2; i++)
            {
                if (xc_func_init(&func1_[i], xc_id[i], XC_UNPOLARIZED) != 0) 
                    error(__FILE__, __LINE__, "functional is not found");
                
                if (xc_func_init(&func2_[i], xc_id[i], XC_POLARIZED) != 0)
                    error(__FILE__, __LINE__, "functional is not found");
            }
        }

        ~libxc_interface()
        {
            for (int i = 0; i < 2; i++)
            {
                xc_func_end(&func1_[i]);
                xc_func_end(&func2_[i]);
            }
        }
        
        void getxc(int size, const double* rho, double* vxc, double* exc)
        {
            memset(vxc, 0, size * sizeof(double)); 
            memset(exc, 0, size * sizeof(double));
    
            std::vector<double> vxc_tmp(size);
            std::vector<double> exc_tmp(size);

            // check rho
            for (int i = 0; i < size; i++)
            {
                if (rho[i] < 0.0)
                {
                    std::stringstream s;
                    s << "rho is negative : " << rho[i];
                    error(__FILE__, __LINE__, s);
                }
            }

            for (int i = 0; i < 2; i++)
            {
                xc_lda_exc_vxc(&func1_[i], size, &rho[0], &exc_tmp[0], &vxc_tmp[0]);
       
                for (int j = 0; j < size; j++)
                {
                    vxc[j] += vxc_tmp[j];
                    exc[j] += exc_tmp[j];
                }
            }
        }

        void getxc(int size, const double* rho, const double* mag, double* vxc, double* bxc, double* exc)
        {
            memset(vxc, 0, size * sizeof(double)); 
            memset(bxc, 0, size * sizeof(double)); 
            memset(exc, 0, size * sizeof(double));

            std::vector<double> rhoud(size * 2);
            for (int i = 0; i < size; i++)
            {
                rhoud[2 * i] = 0.5 * (rho[i] + mag[i]);
                rhoud[2 * i + 1] = 0.5 * (rho[i] - mag[i]);

                if (rhoud[2 * i] < 0.0) error(__FILE__, __LINE__, "rho_up is negative");

                if (rhoud[2 * i + 1] < 0.0)
                {
                    if (fabs(rhoud[2 * i + 1]) > 1e-8)
                    {
                        std::stringstream s;
                        s << "rho_dn is negative : " << rhoud[2 * i + 1] << std::endl
                          << "  rho : " << rho[i] << "   mag : " << mag[i];
                        error(__FILE__, __LINE__, s);
                    }
                    else
                    {
                        rhoud[2 * i + 1] = 0.0;
                    }
                }
            }

            std::vector<double> vxc_tmp(size * 2);
            std::vector<double> exc_tmp(size);
            
            for (int i = 0; i < 2; i++)
            {
                xc_lda_exc_vxc(&func2_[i], size, &rhoud[0], &exc_tmp[0], &vxc_tmp[0]);

                for (int j = 0; j < size; j++)
                {
                    exc[j] += exc_tmp[j];
                    vxc[j] += 0.5 * (vxc_tmp[2 * j] + vxc_tmp[2 * j + 1]);
                    bxc[j] += 0.5 * (vxc_tmp[2 * j] - vxc_tmp[2 * j + 1]);
                }
            }

            for (int i = 0; i < size; i++)
            {
                if (vxc[i] > 0.0) error(__FILE__, __LINE__, "vxc > 0");
                
                if (bxc[i] > 0.0)
                {
                    //if (bxc[i] > 1e-7)
                    //{
                    //    std::stringstream s;
                    //    s << "bxc is positive : " << bxc[i];
                    //    error(__FILE__, __LINE__, s);
                    //}
                    //else
                    //{
                        bxc[i] = 0.0;
                    //}
                }
            }
         }
};

};
