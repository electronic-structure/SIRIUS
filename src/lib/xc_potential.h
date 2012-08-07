namespace sirius
{

class xc_potential
{
    public:
        
        static void get(int size, 
                        double* rho, 
                        double* vxc, 
                        double* exc)
        {
            std::vector<double> tmp(size);
            memset(vxc, 0, size * sizeof(double)); 
            memset(exc, 0, size * sizeof(double));

            int xc_id[] = {1, 7};
            xc_func_type func;
    
            for (int i = 0; i < 2; i++)
            {
                if(xc_func_init(&func, xc_id[i], XC_UNPOLARIZED) != 0)
                    error(__FILE__, __LINE__, "functional is not found");
       
                xc_lda_vxc(&func, size, rho, &tmp[0]);

                for (int j = 0; j < size; j++)
                    vxc[j] += tmp[j];

                xc_lda_exc(&func, size, rho, &tmp[0]);

                for (int j = 0; j < size; j++)
                    exc[j] += tmp[j];
      
                xc_func_end(&func);
            }
        }
};

};
