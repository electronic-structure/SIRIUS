template<processing_unit_t> 
class blas;

// CPU
template<> 
class blas<cpu>
{
    public:

        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T alpha, T* a, int32_t lda, 
                         T* b, int32_t ldb, T beta, T* c, int32_t ldc);
        
        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T* a, int32_t lda, 
                         T* b, int32_t ldb, T* c, int32_t ldc);
                         
        template<typename T>
        static void hemm(int side, int uplo, int32_t m, int32_t n, T alpha, T* a, int32_t lda, 
                         T* b, int32_t ldb, T beta, T* c, int32_t ldc);

        template<typename T>
        static void gemv(int trans, int32_t m, int32_t n, T alpha, T* a, int32_t lda, T* x, int32_t incx, 
                         T beta, T* y, int32_t incy);

        /// generic interface to p?gemm
        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T alpha, 
                         dmatrix<T>& a, int32_t ia, int32_t ja, dmatrix<T>& b, int32_t ib, int32_t jb, T beta, 
                         dmatrix<T>& c, int32_t ic, int32_t jc);

        /// simple interface to p?gemm: all matrices start form (0, 0) corner
        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                         T alpha, dmatrix<T>& a, dmatrix<T>& b, T beta, dmatrix<T>& c);
};

#ifdef _GPU_
template<> 
class blas<gpu>
{
    private:
        static double_complex zone;
        static double_complex zzero;

    public:

        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T* alpha, T* a, int32_t lda, 
                         T* b, int32_t ldb, T* beta, T* c, int32_t ldc);
        
        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T* a, int32_t lda, 
                         T* b, int32_t ldb, T* c, int32_t ldc);
};
#endif
