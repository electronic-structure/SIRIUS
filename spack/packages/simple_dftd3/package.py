# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.build_systems.meson import MesonPackage

from spack.package import *


class SimpleDftd3(MesonPackage):
    """
    Simple reimplementation of the DFT-D3 dispersion correction
    """

    homepage = "https://dftd3.readthedocs.io"
    url = "https://github.com/dftd3/simple-dftd3/archive/refs/tags/v1.2.1.tar.gz"
    git = "https://github.com/dftd3/simple-dftd3.git"

    maintainers("awvwgk")

    license("LGPL-3.0-or-later")

    version("main", branch="main")
    version("1.2.1", sha256="3a12c04c490badc63054aca18ea7670d416fcc2152cfe9b8af220da57c39f942")
    version("1.2.0", sha256="20adc61ed606e71227a78308a9ddca34d822d46117e6311ed51a00df16b2eabc")
    version("1.1.1", sha256="fde5e1bdac41c38692bfdb6abcad66fb9ccfe6e990a8d4cf54f44e7188d49b5a")
    version("1.1.0", sha256="52e43c7d860c8742876baab59d0e93e5c91963d1fecf11fd218655c6740281b1")
    version("1.0.0", sha256="fac3d9f785562b178dcf8e89f8d27782b8bda45fcd9dbaccc359b5def4fb1cf6")
    version("0.7.0", sha256="19400a176eb4dcee7b89181a5a5f0033fe6b05c52821e54896a98448761d003a")
    version("0.6.0", sha256="4bef311f8e5a2c32141eddeea65615c3c8480f917cd884488ede059fb0962a50")
    version("0.5.1", sha256="3d775608bf85cd389385a84ea5586ede57215ff9cff646480552ca835a9de9ca")

    variant("openmp", default=True, description="Use OpenMP parallelisation")
    variant("python", default=False, description="Build Python extension module")

    depends_on("c", type="build")  # generated
    depends_on("fortran", type="build")  # generated

    depends_on("mctc-lib")
    depends_on("meson@0.57.1:", type="build")  # mesonbuild/meson#8377
    depends_on("pkgconfig", type="build")
    depends_on("toml-f")
    depends_on("py-cffi", when="+python")
    depends_on("python@3.6:", when="+python")

    extends("python", when="+python")

    def meson_args(self):
        return [
            "-Dopenmp={0}".format(str("+openmp" in self.spec).lower()),
            "-Dpython={0}".format(str("+python" in self.spec).lower()),
        ]
