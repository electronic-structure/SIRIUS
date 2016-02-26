#include "config-f90.h"

module elpa_mpi
  use precision
#ifndef WITH_MPI
  use elpa_mpi_stubs
#else
  implicit none
  public
  include "mpif.h"
#endif

end module
