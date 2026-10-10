# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.build_systems import cmake, meson
from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.meson import MesonPackage

from spack.package import *


class SimpleDftd3(MesonPackage, CMakePackage):
    """
    Simple reimplementation of the DFT-D3 dispersion correction
    """

    homepage = "https://dftd3.readthedocs.io"
    url = "https://github.com/dftd3/simple-dftd3/archive/refs/tags/v1.2.1.tar.gz"
    git = "https://github.com/dftd3/simple-dftd3.git"

    maintainers("awvwgk")

    license("LGPL-3.0-or-later")

    build_system("cmake", "meson", default="meson")

    version("main", branch="main")
    version("1.6.0", sha256="534919ebc44bb59de323513f41d1c1fba5f457008323d2e67087351a76c8fd9f")
    version("1.5.0", sha256="64a3cb126acddb0aaffd87b1e54f35eca0c30f63bc675cdb8ad523373c7c396f")
    version("1.4.0", sha256="c548629115c3d5f180d06a70bc29dcf42e4018fbc9e4ba7c99abc1cdbfda7c1e")
    version("1.3.2", sha256="bbf6aaa23332a7217a6dfc5c3ca8dc74f949c4cb6e4dccbadf32fa21fe8bb0d7")
    version("1.3.1", sha256="1a49cf2140e1d2e6f954a711b02b7b78e31595480ffa2393d9fa34ab9ea1cfef")
    version("1.3.0", sha256="2892f6c766c9129e6a734264dc9e827491ec398d59634cb905286ea3eaa17b55")
    version("1.2.1", sha256="3a12c04c490badc63054aca18ea7670d416fcc2152cfe9b8af220da57c39f942")
    version("1.2.0", sha256="20adc61ed606e71227a78308a9ddca34d822d46117e6311ed51a00df16b2eabc")
    version("1.1.1", sha256="fde5e1bdac41c38692bfdb6abcad66fb9ccfe6e990a8d4cf54f44e7188d49b5a")
    version("1.1.0", sha256="52e43c7d860c8742876baab59d0e93e5c91963d1fecf11fd218655c6740281b1")
    version("1.0.0", sha256="fac3d9f785562b178dcf8e89f8d27782b8bda45fcd9dbaccc359b5def4fb1cf6")
    version("0.7.0", sha256="19400a176eb4dcee7b89181a5a5f0033fe6b05c52821e54896a98448761d003a")
    version("0.6.0", sha256="4bef311f8e5a2c32141eddeea65615c3c8480f917cd884488ede059fb0962a50")
    version("0.5.1", sha256="3d775608bf85cd389385a84ea5586ede57215ff9cff646480552ca835a9de9ca")

    variant("shared", default=True, description="Build shared libraries")
    variant("openmp", default=True, description="Use OpenMP parallelisation")
    variant(
        "python",
        default=False,
        when="build_system=meson",
        description="Build Python extension module",
    )

    depends_on("c", type="build")  # generated
    depends_on("fortran", type="build")  # generated
    depends_on("meson@0.57.1:", type="build", when="build_system=meson")  # mesonbuild/meson#8377
    depends_on("py-cffi", when="+python")
    depends_on("py-numpy", when="+python")
    depends_on("py-setuptools", type="build", when="+python")
    depends_on("python@3.6:", when="+python")
    depends_on("pkgconfig", type="build")
    depends_on("blas")

    for build_system in ["cmake", "meson"]:
        depends_on(f"mctc-lib build_system={build_system}", when=f"build_system={build_system}")
        depends_on(f"toml-f build_system={build_system}", when=f"build_system={build_system}")
        depends_on(f"mstore build_system={build_system}", when=f"build_system={build_system}")

    extends("python", when="+python")


class MesonBuilder(meson.MesonBuilder):
    def meson_args(self):
        return [
            "-Ddefault_library={0}".format("shared" if "+shared" in self.spec else "static"),
            "-Dopenmp={0}".format(str("+openmp" in self.spec).lower()),
            "-Dpython={0}".format(str("+python" in self.spec).lower()),
        ]


class CMakeBuilder(cmake.CMakeBuilder):
    def cmake_args(self):
        return [
            self.define_from_variant("BUILD_SHARED_LIBS", "shared"),
            self.define_from_variant("WITH_OpenMP", "openmp"),
        ]
