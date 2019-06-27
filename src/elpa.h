/** \file elpa.h
 *
 *  \brief Interface to ELPA library.
 */

using elpa_t = void*;

int elpa_init(int);
int elpa_uninit(int*);

elpa_t elpa_allocate(int *error);
void elpa_deallocate(elpa_t handle, int *error);
int elpa_setup(elpa_t handle);
void elpa_set_integer(elpa_t handle, const char *name, int value, int *error);
void elpa_eigenvectors_d(elpa_t handle, double *a, double *ev, double *q, int *error);
void elpa_eigenvectors_f(elpa_t handle, float *a, float *ev, float *q, int *error);
void elpa_eigenvectors_dc(elpa_t handle, std::complex<double> *a, double *ev, std::complex<double> *q, int *error);
void elpa_eigenvectors_fc(elpa_t handle, std::complex<float> *a, float *ev, std::complex<float> *q, int *error);

 /*! \brief C interface to driver function "elpa_solve_evp_real_double"
 *
 *  \param  na                        Order of matrix a
 *  \param  nev                       Number of eigenvalues needed.
 *                                    The smallest nev eigenvalues/eigenvectors are calculated.
 *  \param  a                         Distributed matrix for which eigenvalues are to be computed.
 *                                    Distribution is like in Scalapack.
 *                                    The full matrix must be set (not only one half like in scalapack).
 *  \param lda                        Leading dimension of a
 *  \param ev(na)                     On output: eigenvalues of a, every processor gets the complete set
 *  \param q                          On output: Eigenvectors of a
 *                                    Distribution is like in Scalapack.
 *                                    Must be always dimensioned to the full size (corresponding to (na,na))
 *                                    even if only a part of the eigenvalues is needed.
 *  \param ldq                        Leading dimension of q
 *  \param nblk                       blocksize of cyclic distribution, must be the same in both directions!
 *  \param matrixCols                 distributed number of matrix columns
 *  \param mpi_comm_rows              MPI-Communicator for rows
 *  \param mpi_comm_cols              MPI-Communicator for columns
 *  \param mpi_coll_all               MPI communicator for the total processor set
 *  \param THIS_REAL_ELPA_KERNEL_API  specify used ELPA2 kernel via API
 *  \param useQR                      use QR decomposition 1 = yes, 0 = no
 *  \param useGPU                     use GPU (1=yes, 0=No)
 *  \param method                     choose whether to use ELPA 1stage or 2stage solver
 *                                    possible values: "1stage" => use ELPA 1stage solver
 *                                                      "2stage" => use ELPA 2stage solver
 *                                                       "auto"   => (at the moment) use ELPA 2stage solver
 *
 *  \result                     int: 1 if error occured, otherwise 0
 */
 int elpa_solve_evp_real_double(int na, int nev, double *a, int lda, double *ev, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_REAL_ELPA_KERNEL_API, int useQR, int useGPU, char *method);
 /*! \brief C interface to driver function "elpa_solve_evp_real_single"
 *
 *  \param  na                        Order of matrix a
 *  \param  nev                       Number of eigenvalues needed.
 *                                    The smallest nev eigenvalues/eigenvectors are calculated.
 *  \param  a                         Distributed matrix for which eigenvalues are to be computed.
 *                                    Distribution is like in Scalapack.
 *                                    The full matrix must be set (not only one half like in scalapack).
 *  \param lda                        Leading dimension of a
 *  \param ev(na)                     On output: eigenvalues of a, every processor gets the complete set
 *  \param q                          On output: Eigenvectors of a
 *                                    Distribution is like in Scalapack.
 *                                    Must be always dimensioned to the full size (corresponding to (na,na))
 *                                    even if only a part of the eigenvalues is needed.
 *  \param ldq                        Leading dimension of q
 *  \param nblk                       blocksize of cyclic distribution, must be the same in both directions!
 *  \param matrixCols                 distributed number of matrix columns
 *  \param mpi_comm_rows              MPI-Communicator for rows
 *  \param mpi_comm_cols              MPI-Communicator for columns
 *  \param mpi_coll_all               MPI communicator for the total processor set
 *  \param THIS_REAL_ELPA_KERNEL_API  specify used ELPA2 kernel via API
 *  \param useQR                      use QR decomposition 1 = yes, 0 = no
 *  \param useGPU                     use GPU (1=yes, 0=No)
 *  \param method                     choose whether to use ELPA 1stage or 2stage solver
 *                                    possible values: "1stage" => use ELPA 1stage solver
 *                                                      "2stage" => use ELPA 2stage solver
 *                                                       "auto"   => (at the moment) use ELPA 2stage solver
 *
 *  \result                     int: 1 if error occured, otherwise 0
 */
 int elpa_solve_evp_real_single(int na, int nev, float *a, int lda, float *ev, float *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_REAL_ELPA_KERNEL_API, int useQR, int useGPU, char *method);
 /*! \brief C interface to driver function "elpa_solve_evp_complex_double"
 *
 *  \param  na                           Order of matrix a
 *  \param  nev                          Number of eigenvalues needed.
 *                                       The smallest nev eigenvalues/eigenvectors are calculated.
 *  \param  a                            Distributed matrix for which eigenvalues are to be computed.
 *                                       Distribution is like in Scalapack.
 *                                       The full matrix must be set (not only one half like in scalapack).
 *  \param lda                           Leading dimension of a
 *  \param ev(na)                        On output: eigenvalues of a, every processor gets the complete set
 *  \param q                             On output: Eigenvectors of a
 *                                       Distribution is like in Scalapack.
 *                                       Must be always dimensioned to the full size (corresponding to (na,na))
 *                                       even if only a part of the eigenvalues is needed.
 *  \param ldq                           Leading dimension of q
 *  \param nblk                          blocksize of cyclic distribution, must be the same in both directions!
 *  \param matrixCols                    distributed number of matrix columns
 *  \param mpi_comm_rows                 MPI-Communicator for rows
 *  \param mpi_comm_cols                 MPI-Communicator for columns
 *  \param mpi_coll_all                  MPI communicator for the total processor set
 *  \param THIS_COMPLEX_ELPA_KERNEL_API  specify used ELPA2 kernel via API
 *  \param useGPU                        use GPU (1=yes, 0=No)
 *  \param method                        choose whether to use ELPA 1stage or 2stage solver
 *                                       possible values: "1stage" => use ELPA 1stage solver
 *                                                        "2stage" => use ELPA 2stage solver
 *                                                         "auto"   => (at the moment) use ELPA 2stage solver
 *
 *  \result                     int: 1 if error occured, otherwise 0
 */
 int elpa_solve_evp_complex_double(int na, int nev, std::complex<double> *a, int lda, double *ev, std::complex<double> *q, int ldq, int nblk, int matrixCols,
                                   int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_COMPLEX_ELPA_KERNEL_API, int useGPU, char *method);
 /*! \brief C interface to driver function "elpa_solve_evp_complex_single"
 *
 *  \param  na                           Order of matrix a
 *  \param  nev                          Number of eigenvalues needed.
 *                                       The smallest nev eigenvalues/eigenvectors are calculated.
 *  \param  a                            Distributed matrix for which eigenvalues are to be computed.
 *                                       Distribution is like in Scalapack.
 *                                       The full matrix must be set (not only one half like in scalapack).
 *  \param lda                           Leading dimension of a
 *  \param ev(na)                        On output: eigenvalues of a, every processor gets the complete set
 *  \param q                             On output: Eigenvectors of a
 *                                       Distribution is like in Scalapack.
 *                                       Must be always dimensioned to the full size (corresponding to (na,na))
 *                                       even if only a part of the eigenvalues is needed.
 *  \param ldq                           Leading dimension of q
 *  \param nblk                          blocksize of cyclic distribution, must be the same in both directions!
 *  \param matrixCols                    distributed number of matrix columns
 *  \param mpi_comm_rows                 MPI-Communicator for rows
 *  \param mpi_comm_cols                 MPI-Communicator for columns
 *  \param mpi_coll_all                  MPI communicator for the total processor set
 *  \param THIS_COMPLEX_ELPA_KERNEL_API  specify used ELPA2 kernel via API
 *  \param useGPU                        use GPU (1=yes, 0=No)
 *  \param method                        choose whether to use ELPA 1stage or 2stage solver
 *                                       possible values: "1stage" => use ELPA 1stage solver
 *                                                        "2stage" => use ELPA 2stage solver
 *                                                         "auto"   => (at the moment) use ELPA 2stage solver
 *
 *  \result                     int: 1 if error occured, otherwise 0
 */
 int elpa_solve_evp_complex_single(int na, int nev, std::complex<float> *a, int lda, float *ev, std::complex<float> *q, int ldq, int nblk, int matrixCols,
                                   int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_COMPLEX_ELPA_KERNEL_API, int useGPU, char *method);
 /*! \brief C old, deprecated interface, will be deleted. Use "elpa_get_communicators"
 *
 * \param mpi_comm_word    MPI global communicator (in)
 * \param my_prow          Row coordinate of the calling process in the process grid (in)
 * \param my_pcol          Column coordinate of the calling process in the process grid (in)
 * \param mpi_comm_rows    Communicator for communicating within rows of processes (out)
 * \result int             integer error value of mpi_comm_split function
 */
 int get_elpa_row_col_comms(int mpi_comm_world, int my_prow, int my_pcol, int *mpi_comm_rows, int *mpi_comm_cols);
 /*! \brief C old, deprecated interface, will be deleted. Use "elpa_get_communicators"
 *
 * \param mpi_comm_word    MPI global communicator (in)
 * \param my_prow          Row coordinate of the calling process in the process grid (in)
 * \param my_pcol          Column coordinate of the calling process in the process grid (in)
 * \param mpi_comm_rows    Communicator for communicating within rows of processes (out)
 * \result int             integer error value of mpi_comm_split function
 */
 int get_elpa_communicators(int mpi_comm_world, int my_prow, int my_pcol, int *mpi_comm_rows, int *mpi_comm_cols);
 /*! \brief C interface to create ELPA communicators
 *
 * \param mpi_comm_word    MPI global communicator (in)
 * \param my_prow          Row coordinate of the calling process in the process grid (in)
 * \param my_pcol          Column coordinate of the calling process in the process grid (in)
 * \param mpi_comm_rows    Communicator for communicating within rows of processes (out)
 * \result int             integer error value of mpi_comm_split function
 */
 int elpa_get_communicators(int mpi_comm_world, int my_prow, int my_pcol, int *mpi_comm_rows, int *mpi_comm_cols);
  /*! \brief C interface to solve the double-precision real eigenvalue problem with 1-stage solver
  *
 *  \param  na                   Order of matrix a
 *  \param  nev                  Number of eigenvalues needed.
 *                               The smallest nev eigenvalues/eigenvectors are calculated.
 *  \param  a                    Distributed matrix for which eigenvalues are to be computed.
 *                               Distribution is like in Scalapack.
 *                               The full matrix must be set (not only one half like in scalapack).
 *  \param lda                   Leading dimension of a
 *  \param ev(na)                On output: eigenvalues of a, every processor gets the complete set
 *  \param q                     On output: Eigenvectors of a
 *                               Distribution is like in Scalapack.
 *                               Must be always dimensioned to the full size (corresponding to (na,na))
 *                               even if only a part of the eigenvalues is needed.
 *  \param ldq                   Leading dimension of q
 *  \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
 *  \param matrixCols           distributed number of matrix columns
 *  \param mpi_comm_rows        MPI-Communicator for rows
 *  \param mpi_comm_cols        MPI-Communicator for columns
 *  \param useGPU               use GPU (1=yes, 0=No)
 *
 *  \result                     int: 1 if error occured, otherwise 0
*/
 int elpa_solve_evp_real_1stage_double_precision(int na, int nev, double *a, int lda, double *ev, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int useGPU);
  /*! \brief C interface to solve the single-precision real eigenvalue problem with 1-stage solver
  *
 *  \param  na                   Order of matrix a
 *  \param  nev                  Number of eigenvalues needed.
 *                               The smallest nev eigenvalues/eigenvectors are calculated.
 *  \param  a                    Distributed matrix for which eigenvalues are to be computed.
 *                               Distribution is like in Scalapack.
 *                               The full matrix must be set (not only one half like in scalapack).
 *  \param lda                   Leading dimension of a
 *  \param ev(na)                On output: eigenvalues of a, every processor gets the complete set
 *  \param q                     On output: Eigenvectors of a
 *                               Distribution is like in Scalapack.
 *                               Must be always dimensioned to the full size (corresponding to (na,na))
 *                               even if only a part of the eigenvalues is needed.
 *  \param ldq                   Leading dimension of q
 *  \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
 *  \param matrixCols           distributed number of matrix columns
 *  \param mpi_comm_rows        MPI-Communicator for rows
 *  \param mpi_comm_cols        MPI-Communicator for columns
 *  \param useGPU               use GPU (1=yes, 0=No)
 *
 *  \result                     int: 1 if error occured, otherwise 0
*/
 int elpa_solve_evp_real_1stage_single_precision(int na, int nev, float *a, int lda, float *ev, float *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int useGPU);
 /*! \brief C interface to solve the double-precision complex eigenvalue problem with 1-stage solver
 *
 *  \param  na                   Order of matrix a
 *  \param  nev                  Number of eigenvalues needed.
 *                               The smallest nev eigenvalues/eigenvectors are calculated.
 *  \param  a                    Distributed matrix for which eigenvalues are to be computed.
 *                               Distribution is like in Scalapack.
 *                               The full matrix must be set (not only one half like in scalapack).
 *  \param lda                   Leading dimension of a
 *  \param ev(na)                On output: eigenvalues of a, every processor gets the complete set
 *  \param q                     On output: Eigenvectors of a
 *                               Distribution is like in Scalapack.
 *                               Must be always dimensioned to the full size (corresponding to (na,na))
 *                               even if only a part of the eigenvalues is needed.
 *  \param ldq                   Leading dimension of q
 *  \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
 *  \param matrixCols           distributed number of matrix columns
 *  \param mpi_comm_rows        MPI-Communicator for rows
 *  \param mpi_comm_cols        MPI-Communicator for columns
 *  \param useGPU               use GPU (1=yes, 0=No)
 *
 *  \result                     int: 1 if error occured, otherwise 0
 */
 int elpa_solve_evp_complex_1stage_double_precision(int na, int nev, std::complex<double> *a, int lda, double *ev, std::complex<double> *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int useGPU);
 /*! \brief C interface to solve the single-precision complex eigenvalue problem with 1-stage solver
 *
 *  \param  na                   Order of matrix a
 *  \param  nev                  Number of eigenvalues needed.
 *                               The smallest nev eigenvalues/eigenvectors are calculated.
 *  \param  a                    Distributed matrix for which eigenvalues are to be computed.
 *                               Distribution is like in Scalapack.
 *                               The full matrix must be set (not only one half like in scalapack).
 *  \param lda                   Leading dimension of a
 *  \param ev(na)                On output: eigenvalues of a, every processor gets the complete set
 *  \param q                     On output: Eigenvectors of a
 *                               Distribution is like in Scalapack.
 *                               Must be always dimensioned to the full size (corresponding to (na,na))
 *                               even if only a part of the eigenvalues is needed.
 *  \param ldq                   Leading dimension of q
 *  \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
 *  \param matrixCols           distributed number of matrix columns
 *  \param mpi_comm_rows        MPI-Communicator for rows
 *  \param mpi_comm_cols        MPI-Communicator for columns
 *  \param useGPU               use GPU (1=yes, 0=No)
 *
 *  \result                     int: 1 if error occured, otherwise 0
 */
 int elpa_solve_evp_complex_1stage_single_precision(int na, int nev,  std::complex<float> *a, int lda, float *ev, std::complex<float> *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int useGPU);
 /*
 \brief  C interface to solve double-precision tridiagonal eigensystem with divide and conquer method
 \details

 *\param na                    Matrix dimension
 *\param nev                   number of eigenvalues/vectors to be computed
 *\param d                     array d(na) on input diagonal elements of tridiagonal matrix, on
 *                             output the eigenvalues in ascending order
 *\param e                     array e(na) on input subdiagonal elements of matrix, on exit destroyed
 *\param q                     on exit : matrix q(ldq,matrixCols) contains the eigenvectors
 *\param ldq                   leading dimension of matrix q
 *\param nblk                  blocksize of cyclic distribution, must be the same in both directions!
 *\param matrixCols            columns of matrix q
 *\param mpi_comm_rows         MPI communicator for rows
 *\param mpi_comm_cols         MPI communicator for columns
 *\param wantDebug             give more debug information if 1, else 0
 *\result success              int 1 on success, else 0
 */
 int elpa_solve_tridi_double(int na, int nev, double *d, double *e, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
 /*
 \brief  C interface to solve single-precision tridiagonal eigensystem with divide and conquer method
 \details

 \param na                    Matrix dimension
 \param nev                   number of eigenvalues/vectors to be computed
 \param d                     array d(na) on input diagonal elements of tridiagonal matrix, on
                              output the eigenvalues in ascending order
 \param e                     array e(na) on input subdiagonal elements of matrix, on exit destroyed
 \param q                     on exit : matrix q(ldq,matrixCols) contains the eigenvectors
 \param ldq                   leading dimension of matrix q
 \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
 \param matrixCols            columns of matrix q
 \param mpi_comm_rows         MPI communicator for rows
 \param mpi_comm_cols         MPI communicator for columns
 \param wantDebug             give more debug information if 1, else 0
 \result success              int 1 on success, else 0
 */
 int elpa_solve_tridi_single(int na, int nev, float *d, float *e, float *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
 /*
 \brief  C interface for elpa_mult_at_b_real_double: Performs C : = A**T * B for double-precision matrices
         where   A is a square matrix (na,na) which is optionally upper or lower triangular
                 B is a (na,ncb) matrix
                 C is a (na,ncb) matrix where optionally only the upper or lower
                   triangle may be computed
 \details
 \param  uplo_a               'U' if A is upper triangular
                              'L' if A is lower triangular
                              anything else if A is a full matrix
                              Please note: This pertains to the original A (as set in the calling program)
                                           whereas the transpose of A is used for calculations
                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
                              i.e. it may contain arbitrary numbers
 \param uplo_c                'U' if only the upper diagonal part of C is needed
                              'L' if only the upper diagonal part of C is needed
                              anything else if the full matrix C is needed
                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
                                            written to a certain extent, i.e. one shouldn't rely on the content there!
 \param na                    Number of rows/columns of A, number of rows of B and C
 \param ncb                   Number of columns  of B and C
 \param a                     matrix a
 \param lda                   leading dimension of matrix a
 \param ldaCols               columns of matrix a
 \param b                     matrix b
 \param ldb                   leading dimension of matrix b
 \param ldbCols               columns of matrix b
 \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
 \param  mpi_comm_rows        MPI communicator for rows
 \param  mpi_comm_cols        MPI communicator for columns
 \param c                     matrix c
 \param ldc                   leading dimension of matrix c
 \param ldcCols               columns of matrix c
 \result success              int report success (1) or failure (0)
 */
 int elpa_mult_at_b_real_double(char uplo_a, char uplo_c, int na, int ncb, double *a, int lda, int ldaCols, double *b, int ldb, int ldbCols, int nlbk, int mpi_comm_rows, int mpi_comm_cols, double *c, int ldc, int ldcCols);
 /*
 \brief  C interface for elpa_mult_at_b_real_single: Performs C : = A**T * B for single-precision matrices
         where   A is a square matrix (na,na) which is optionally upper or lower triangular
                 B is a (na,ncb) matrix
                 C is a (na,ncb) matrix where optionally only the upper or lower
                   triangle may be computed
 \details
 \param  uplo_a               'U' if A is upper triangular
                              'L' if A is lower triangular
                              anything else if A is a full matrix
                              Please note: This pertains to the original A (as set in the calling program)
                                           whereas the transpose of A is used for calculations
                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
                              i.e. it may contain arbitrary numbers
 \param uplo_c                'U' if only the upper diagonal part of C is needed
                              'L' if only the upper diagonal part of C is needed
                              anything else if the full matrix C is needed
                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
                                            written to a certain extent, i.e. one shouldn't rely on the content there!
 \param na                    Number of rows/columns of A, number of rows of B and C
 \param ncb                   Number of columns  of B and C
 \param a                     matrix a
 \param lda                   leading dimension of matrix a
 \param ldaCols               columns of matrix a
 \param b                     matrix b
 \param ldb                   leading dimension of matrix b
 \param ldbCols               columns of matrix b
 \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
 \param  mpi_comm_rows        MPI communicator for rows
 \param  mpi_comm_cols        MPI communicator for columns
 \param c                     matrix c
 \param ldc                   leading dimension of matrix c
 \result success              int report success (1) or failure (0)
 */
 int elpa_mult_at_b_real_single(char uplo_a, char uplo_c, int na, int ncb, float *a, int lda, int ldaCols, float *b, int ldb, int ldbCols, int nlbk, int mpi_comm_rows, int mpi_comm_cols, float *c, int ldc, int ldcCols);
 /*
 \brief C interface for elpa_mult_ah_b_complex_double: Performs C : = A**H * B for double-precision matrices
         where   A is a square matrix (na,na) which is optionally upper or lower triangular
                 B is a (na,ncb) matrix
                 C is a (na,ncb) matrix where optionally only the upper or lower
                   triangle may be computed
 \details

 \param  uplo_a               'U' if A is upper triangular
                              'L' if A is lower triangular
                              anything else if A is a full matrix
                              Please note: This pertains to the original A (as set in the calling program)
                                           whereas the transpose of A is used for calculations
                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
                              i.e. it may contain arbitrary numbers
 \param uplo_c                'U' if only the upper diagonal part of C is needed
                              'L' if only the upper diagonal part of C is needed
                              anything else if the full matrix C is needed
                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
                                            written to a certain extent, i.e. one shouldn't rely on the content there!
 \param na                    Number of rows/columns of A, number of rows of B and C
 \param ncb                   Number of columns  of B and C
 \param a                     matrix a
 \param lda                   leading dimension of matrix a
 \param b                     matrix b
 \param ldb                   leading dimension of matrix b
 \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
 \param  mpi_comm_rows        MPI communicator for rows
 \param  mpi_comm_cols        MPI communicator for columns
 \param c                     matrix c
 \param ldc                   leading dimension of matrix c
 \result success              int reports success (1) or failure (0)
 */
 int elpa_mult_ah_b_complex_double(char uplo_a, char uplo_c, int na, int ncb, std::complex<double> *a, int lda, int ldaCols, std::complex<double> *b, int ldb, int ldbCols, int nblk, int mpi_comm_rows, int mpi_comm_cols, std::complex<double> *c, int ldc, int ldcCols);
 /*
 \brief C interface for elpa_mult_ah_b_complex_single: Performs C : = A**H * B for single-precision matrices
         where   A is a square matrix (na,na) which is optionally upper or lower triangular
                 B is a (na,ncb) matrix
                 C is a (na,ncb) matrix where optionally only the upper or lower
                   triangle may be computed
 \details

 \param  uplo_a               'U' if A is upper triangular
                              'L' if A is lower triangular
                              anything else if A is a full matrix
                              Please note: This pertains to the original A (as set in the calling program)
                                           whereas the transpose of A is used for calculations
                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
                              i.e. it may contain arbitrary numbers
 \param uplo_c                'U' if only the upper diagonal part of C is needed
                              'L' if only the upper diagonal part of C is needed
                              anything else if the full matrix C is needed
                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
                                            written to a certain extent, i.e. one shouldn't rely on the content there!
 \param na                    Number of rows/columns of A, number of rows of B and C
 \param ncb                   Number of columns  of B and C
 \param a                     matrix a
 \param lda                   leading dimension of matrix a
 \param b                     matrix b
 \param ldb                   leading dimension of matrix b
 \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
 \param  mpi_comm_rows        MPI communicator for rows
 \param  mpi_comm_cols        MPI communicator for columns
 \param c                     matrix c
 \param ldc                   leading dimension of matrix c
 \result success              int reports success (1) or failure (0)
 */
 int elpa_mult_ah_b_complex_single(char uplo_a, char uplo_c, int na, int ncb, std::complex<float> *a, int lda, int ldaCols, std::complex<float> *b, int ldb, int ldbCols, int nblk, int mpi_comm_rows, int mpi_comm_cols, std::complex<float> *c, int ldc, int ldcCols);
 /*
 \brief  C interface to elpa_invert_trm_real_double: Inverts a real double-precision upper triangular matrix
 \details
 \param  na                   Order of matrix
 \param  a(lda,matrixCols)    Distributed matrix which should be inverted
                              Distribution is like in Scalapack.
                              Only upper triangle is needs to be set.
                              The lower triangle is not referenced.
 \param  lda                  Leading dimension of a
 \param                       matrixCols  local columns of matrix a
 \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 \param  mpi_comm_rows        MPI communicator for rows
 \param  mpi_comm_cols        MPI communicator for columns
 \param wantDebug             int more debug information on failure if 1, else 0
 \result succes               int reports success (1) or failure (0)
 */
 int elpa_invert_trm_real_double(int na, double *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
 /*
 \brief  C interface to elpa_invert_trm_real_single: Inverts a real single-precision upper triangular matrix
 \details
 \param  na                   Order of matrix
 \param  a(lda,matrixCols)    Distributed matrix which should be inverted
                              Distribution is like in Scalapack.
                              Only upper triangle is needs to be set.
                              The lower triangle is not referenced.
 \param  lda                  Leading dimension of a
 \param                       matrixCols  local columns of matrix a
 \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 \param  mpi_comm_rows        MPI communicator for rows
 \param  mpi_comm_cols        MPI communicator for columns
 \param wantDebug             int more debug information on failure if 1, else 0
 \result succes               int reports success (1) or failure (0)
 */
 int elpa_invert_trm_real_single(int na, double *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
 /*
 \brief  C interface to elpa_invert_trm_complex_double: Inverts a double-precision complex upper triangular matrix
 \details
 \param  na                   Order of matrix
 \param  a(lda,matrixCols)    Distributed matrix which should be inverted
                              Distribution is like in Scalapack.
                              Only upper triangle is needs to be set.
                              The lower triangle is not referenced.
 \param  lda                  Leading dimension of a
 \param                       matrixCols  local columns of matrix a
 \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 \param  mpi_comm_rows        MPI communicator for rows
 \param  mpi_comm_cols        MPI communicator for columns
 \param wantDebug             int more debug information on failure if 1, else 0
 \result succes               int reports success (1) or failure (0)
 */
 int elpa_invert_trm_complex_double(int na, std::complex<double> *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
 /*
 \brief  C interface to elpa_invert_trm_complex_single: Inverts a single-precision complex upper triangular matrix
 \details
 \param  na                   Order of matrix
 \param  a(lda,matrixCols)    Distributed matrix which should be inverted
                              Distribution is like in Scalapack.
                              Only upper triangle is needs to be set.
                              The lower triangle is not referenced.
 \param  lda                  Leading dimension of a
 \param                       matrixCols  local columns of matrix a
 \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 \param  mpi_comm_rows        MPI communicator for rows
 \param  mpi_comm_cols        MPI communicator for columns
 \param wantDebug             int more debug information on failure if 1, else 0
 \result succes               int reports success (1) or failure (0)
 */
 int elpa_invert_trm_complex_single(int na, std::complex<float> *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
 /*
 \brief  elpa_cholesky_real_double: Cholesky factorization of a double-precision real symmetric matrix
 \details

 *\param  na                   Order of matrix
 *\param  a(lda,matrixCols)    Distributed matrix which should be factorized.
 *                             Distribution is like in Scalapack.
 *                             Only upper triangle is needs to be set.
 *                             On return, the upper triangle contains the Cholesky factor
 *                             and the lower triangle is set to 0.
 *\param  lda                  Leading dimension of a
 *\param  matrixCols           local columns of matrix a
 *\param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 *\param  mpi_comm_rows        MPI communicator for rows
 *\param  mpi_comm_cols        MPI communicator for columns
 *\param wantDebug             int more debug information on failure if 1, else 0
 *\result succes               int reports success (1) or failure (0)
 */
 int elpa_cholesky_real_double(int na, double *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
 /*
 \brief  elpa_cholesky_real_single: Cholesky factorization of a single-precision real symmetric matrix
 \details

 \param  na                   Order of matrix
 \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
                              Distribution is like in Scalapack.
                              Only upper triangle is needs to be set.
                              On return, the upper triangle contains the Cholesky factor
                              and the lower triangle is set to 0.
 \param  lda                  Leading dimension of a
 \param                       matrixCols  local columns of matrix a
 \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 \param  mpi_comm_rows        MPI communicator for rows
 \param  mpi_comm_cols        MPI communicator for columns
 \param wantDebug             int more debug information on failure if 1, else 0
 \result succes               int reports success (1) or failure (0)
 */
 int elpa_cholesky_real_single(int na, float *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
 /*
 \brief  C interface elpa_cholesky_complex_double: Cholesky factorization of a double-precision complex hermitian matrix
 \details
 \param  na                   Order of matrix
 \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
                              Distribution is like in Scalapack.
                              Only upper triangle is needs to be set.
                              On return, the upper triangle contains the Cholesky factor
                              and the lower triangle is set to 0.
 \param  lda                  Leading dimension of a
 \param                       matrixCols  local columns of matrix a
 \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 \param  mpi_comm_rows        MPI communicator for rows
 \param  mpi_comm_cols        MPI communicator for columns
 \param wantDebug             int more debug information on failure, if 1, else 0
 \result succes               int reports success (1) or failure (0)
 */
 int elpa_cholesky_complex_double(int na, std::complex<double> *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
 /*
 \brief  C interface elpa_cholesky_complex_single: Cholesky factorization of a single-precision complex hermitian matrix
 \details
 \param  na                   Order of matrix
 \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
                              Distribution is like in Scalapack.
                              Only upper triangle is needs to be set.
                              On return, the upper triangle contains the Cholesky factor
                              and the lower triangle is set to 0.
 \param  lda                  Leading dimension of a
 \param                       matrixCols  local columns of matrix a
 \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 \param  mpi_comm_rows        MPI communicator for rows
 \param  mpi_comm_cols        MPI communicator for columns
 \param wantDebug             int more debug information on failure, if 1, else 0
 \result succes               int reports success (1) or failure (0)
 */
 int elpa_cholesky_complex_single(int na, std::complex<float> *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
 /*! \brief C interface to solve the double-precision real eigenvalue problem with 2-stage solver
 *
 *  \param  na                        Order of matrix a
 *  \param  nev                       Number of eigenvalues needed.
 *                                    The smallest nev eigenvalues/eigenvectors are calculated.
 *  \param  a                         Distributed matrix for which eigenvalues are to be computed.
 *                                    Distribution is like in Scalapack.
 *                                    The full matrix must be set (not only one half like in scalapack).
 *  \param lda                        Leading dimension of a
 *  \param ev(na)                     On output: eigenvalues of a, every processor gets the complete set
 *  \param q                          On output: Eigenvectors of a
 *                                    Distribution is like in Scalapack.
 *                                    Must be always dimensioned to the full size (corresponding to (na,na))
 *                                    even if only a part of the eigenvalues is needed.
 *  \param ldq                        Leading dimension of q
 *  \param nblk                       blocksize of cyclic distribution, must be the same in both directions!
 *  \param matrixCols                 distributed number of matrix columns
 *  \param mpi_comm_rows              MPI-Communicator for rows
 *  \param mpi_comm_cols              MPI-Communicator for columns
 *  \param mpi_coll_all               MPI communicator for the total processor set
 *  \param THIS_REAL_ELPA_KERNEL_API  specify used ELPA2 kernel via API
 *  \param useQR                      use QR decomposition 1 = yes, 0 = no
 *  \param useGPU                     use GPU (1=yes, 0=No)
 *
 *  \result                     int: 1 if error occured, otherwise 0
 */
 int elpa_solve_evp_real_2stage_double_precision(int na, int nev, double *a, int lda, double *ev, double *q, int ldq, int nblk, 
 int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_REAL_ELPA_KERNEL_API, int useQR, int useGPU);
 /*! \brief C interface to solve the single-precision real eigenvalue problem with 2-stage solver
 *
 *  \param  na                        Order of matrix a
 *  \param  nev                       Number of eigenvalues needed.
 *                                    The smallest nev eigenvalues/eigenvectors are calculated.
 *  \param  a                         Distributed matrix for which eigenvalues are to be computed.
 *                                    Distribution is like in Scalapack.
 *                                    The full matrix must be set (not only one half like in scalapack).
 *  \param lda                        Leading dimension of a
 *  \param ev(na)                     On output: eigenvalues of a, every processor gets the complete set
 *  \param q                          On output: Eigenvectors of a
 *                                    Distribution is like in Scalapack.
 *                                    Must be always dimensioned to the full size (corresponding to (na,na))
 *                                    even if only a part of the eigenvalues is needed.
 *  \param ldq                        Leading dimension of q
 *  \param nblk                       blocksize of cyclic distribution, must be the same in both directions!
 *  \param matrixCols                 distributed number of matrix columns
 *  \param mpi_comm_rows              MPI-Communicator for rows
 *  \param mpi_comm_cols              MPI-Communicator for columns
 *  \param mpi_coll_all               MPI communicator for the total processor set
 *  \param THIS_REAL_ELPA_KERNEL_API  specify used ELPA2 kernel via API
 *  \param useQR                      use QR decomposition 1 = yes, 0 = no
 *  \param useGPU                     use GPU (1=yes, 0=No)
 *
 *  \result                     int: 1 if error occured, otherwise 0
 */
 int elpa_solve_evp_real_2stage_single_precision(int na, int nev, float *a, int lda, float *ev, float *q, int ldq, int nblk, 
 int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_REAL_ELPA_KERNEL_API, int useQR, int useGPU);
 /*! \brief C interface to solve the double-precision complex eigenvalue problem with 2-stage solver
 *
 *  \param  na                        Order of matrix a
 *  \param  nev                       Number of eigenvalues needed.
 *                                    The smallest nev eigenvalues/eigenvectors are calculated.
 *  \param  a                         Distributed matrix for which eigenvalues are to be computed.
 *                                    Distribution is like in Scalapack.
 *                                    The full matrix must be set (not only one half like in scalapack).
 *  \param lda                        Leading dimension of a
 *  \param ev(na)                     On output: eigenvalues of a, every processor gets the complete set
 *  \param q                          On output: Eigenvectors of a
 *                                    Distribution is like in Scalapack.
 *                                    Must be always dimensioned to the full size (corresponding to (na,na))
 *                                    even if only a part of the eigenvalues is needed.
 *  \param ldq                        Leading dimension of q
 *  \param nblk                       blocksize of cyclic distribution, must be the same in both directions!
 *  \param matrixCols                 distributed number of matrix columns
 *  \param mpi_comm_rows              MPI-Communicator for rows
 *  \param mpi_comm_cols              MPI-Communicator for columns
 *  \param mpi_coll_all               MPI communicator for the total processor set
 *  \param THIS_COMPLEX_ELPA_KERNEL_API  specify used ELPA2 kernel via API
 *  \param useGPU                     use GPU (1=yes, 0=No)
 *
 *  \result                     int: 1 if error occured, otherwise 0
 */
 int elpa_solve_evp_complex_2stage_double_precision(int na, int nev, std::complex<double> *a, int lda, double *ev, std::complex<double> *q, int ldq, 
 int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_COMPLEX_ELPA_KERNEL_API, int useGPU);
 /*! \brief C interface to solve the single-precision complex eigenvalue problem with 2-stage solver
 *
 *  \param  na                        Order of matrix a
 *  \param  nev                       Number of eigenvalues needed.
 *                                    The smallest nev eigenvalues/eigenvectors are calculated.
 *  \param  a                         Distributed matrix for which eigenvalues are to be computed.
 *                                    Distribution is like in Scalapack.
 *                                    The full matrix must be set (not only one half like in scalapack).
 *  \param lda                        Leading dimension of a
 *  \param ev(na)                     On output: eigenvalues of a, every processor gets the complete set
 *  \param q                          On output: Eigenvectors of a
 *                                    Distribution is like in Scalapack.
 *                                    Must be always dimensioned to the full size (corresponding to (na,na))
 *                                    even if only a part of the eigenvalues is needed.
 *  \param ldq                        Leading dimension of q
 *  \param nblk                       blocksize of cyclic distribution, must be the same in both directions!
 *  \param matrixCols                 distributed number of matrix columns
 *  \param mpi_comm_rows              MPI-Communicator for rows
 *  \param mpi_comm_cols              MPI-Communicator for columns
 *  \param mpi_coll_all               MPI communicator for the total processor set
 *  \param THIS_REAL_ELPA_KERNEL_API  specify used ELPA2 kernel via API
 *  \param useGPU                     use GPU (1=yes, 0=No)
 *
 *  \result                     int: 1 if error occured, otherwise 0
 */
 int elpa_solve_evp_complex_2stage_single_precision(int na, int nev, std::complex<float> *a, int lda, float *ev, std::complex<float> *q, int ldq, int nblk, 
 int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_COMPLEX_ELPA_KERNEL_API, int useGPU);
