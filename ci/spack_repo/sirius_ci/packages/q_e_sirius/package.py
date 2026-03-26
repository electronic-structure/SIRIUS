# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
# adapted from official quantum espresso package


from spack_repo.builtin.build_systems.cmake import CMakePackage

from spack.package import *


class QESirius(CMakePackage):
    """SIRIUS enabled fork of QuantumESPRESSO."""

    homepage = "https://github.com/electronic-structure/q-e-sirius/"
    url = "https://github.com/electronic-structure/q-e-sirius/archive/refs/tags/q-e-sirius/1.0.0.tar.gz"
    git = "https://github.com/electronic-structure/q-e-sirius.git"

    maintainers("simonpintarelli")

    license("GPL-2.0-or-later")

    version("develop-ristretto", branch="ristretto", submodules=True)
    version("1.0.2", sha256="6885979d6c23b81b49d4a96c7f73f2eac618adccb0181bfb96ae2318664d9828")
    version("1.0.1", sha256="512f982aa60fe9fd1cc588fa270e74427c66b62cb2d02ac1cb6cd07dcbe72204")
    version("1.0.0", sha256="d85485db8e9252a0bcd67a6a348b2a74626030183199b0edeb97f14c33bca15b")

    variant("openmp", default=True, description="Enables OpenMP support")
    variant("libxc", default=False, description="Support functionals through libxc")
    variant("sirius_apps", default=False, description="Build SIRIUS standalone binaries")
    # Support for HDF5 has been added starting in QE 6.1.0 and is
    # still experimental
    variant(
        "hdf5",
        default="none",
        description="Orbital and density data I/O with HDF5",
        values=("parallel", "serial", "none"),
        multi=False,
    )

    depends_on("fortran", type="build")
    depends_on("c", type="build")
    depends_on("cxx", type="build")

    depends_on("sirius +fortran")
    depends_on("sirius +apps", when="+sirius_apps")
    depends_on("sirius ~apps", when="~sirius_apps")
    depends_on("sirius +openmp", when="+openmp")
    depends_on("sirius@develop", when="@develop-ristretto")

    depends_on("mpi")
    depends_on("elpa", when="+elpa")
    depends_on("libxc", when="+libxc")
    depends_on("fftw-api@3")
    depends_on("blas")
    depends_on("lapack")
    depends_on("git", type="build")
    depends_on("pkgconfig", type="build")

    variant("scalapack", default=True, description="Enables scalapack support")

    with when("+scalapack"):
        depends_on("scalapack")
        variant("elpa", default=False, description="Uses elpa as an eigenvalue solver")

    # Versions of HDF5 prior to 1.8.16 lead to QE runtime errors
    depends_on("hdf5@1.8.16:+fortran+hl+mpi", when="hdf5=parallel")
    depends_on("hdf5@1.8.16:+fortran+hl~mpi", when="hdf5=serial")

    with when("+openmp"):
        requires("^fftw+openmp", when="^[virtuals=fftw-api] fftw")
        requires("^openblas threads=openmp", when="^[virtuals=blas] openblas")
        requires("^intel-oneapi-mkl threads=openmp", when="^[virtuals=blas] intel-oneapi-mkl")

    def cmake_args(self):
        args = [
            "-DQE_ENABLE_SIRIUS=ON",
            "-DQE_ENABLE_CUDA=OFF",
            "-DQE_LAPACK_INTERNAL=OFF",
            "-DQE_ENABLE_DOC=OFF",
            "-DQE_ENABLE_MPI=ON",
            self.define_from_variant("QE_ENABLE_OPENMP", "openmp"),
            self.define_from_variant("QE_ENABLE_ELPA", "elpa"),
            self.define_from_variant("QE_ENABLE_LIBXC", "libxc"),
            self.define_from_variant("QE_ENABLE_SCALAPACK", "scalapack"),
        ]

        if not self.spec.satisfies("hdf5=none"):
            args.append(self.define("QE_ENABLE_HDF5", True))

        # Work around spack issue #19970 where spack sets
        # rpaths for MKL just during make, but cmake removes
        # them during make install.
        if self.spec.satisfies("^[virtuals=lapack] intel-oneapi-mkl"):
            args.append("-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON")
        spec = self.spec
        args.append(self.define("BLAS_LIBRARIES", spec["blas"].libs.joined(";")))
        args.append(self.define("LAPACK_LIBRARIES", spec["lapack"].libs.joined(";")))

        return args
