# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.build_systems import cmake, meson
from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.meson import MesonPackage

from spack.package import *


class TomlF(MesonPackage, CMakePackage):
    """TOML parser implementation for data serialization and deserialization in Fortran"""

    homepage = "https://toml-f.readthedocs.io/"
    url = "https://github.com/toml-f/toml-f/releases/download/v0.4.2/toml-f-0.4.2.tar.xz"
    git = "https://github.com/toml-f/toml-f/"

    maintainers("awvwgk", "mtaillefumier")

    license("Apache-2.0")

    build_system("cmake", "meson", default="meson")

    version("main", branch="main")
    version("0.5.2", sha256="e18ccc26ac1e7580ffb22feda0fbafd228473248f71e84eac171a455a7915d59")
    version("0.5.1", sha256="961221000469a03306dbd99baf20644b6071cbda71add123e1cdf4a6183a2e91")
    version("0.5.0", sha256="9a6c129836d093efd0cab5a2a013330020682a82e8f59a3882e9ae3249c8281f")
    version("0.4.3", sha256="0b53cf9ec98eff01a81d956e9ef851c16b3c6b49c934d0665b43db9f6ee39953")
    version("0.4.2", sha256="6b49013d3bcd1043494c140d7b2da6b0cedd87648e4fc5179fcfcf41226d3232")
    version("0.4.1", sha256="a95ef65c7d14c1efa86df3d4755889016b6f16ae67f1b9cee7b7ee4dcbe84560")
    version("0.4.0", sha256="1f0e3a75ab6d4832a60698b40f46e8d91b96c7d2ea3f6389d745438631889ceb")
    version("0.3.1", sha256="7f40f60c8d9ffbb1b99fb051a3e6682c7dd04d7479aa1cf770eff8174b02544f")
    version("0.3.0", sha256="40ceca008091607165a09961b79312abfdbbda71cbb94a9dc2625b88c93ff45a")
    version("0.2.4", sha256="ebfeb1e201725b98bae3e656bde4eea2db90154efa8681de758f1389fec902cf")
    version("0.2.3", sha256="2dca7ff6d3e35415cd92454c31560d2b656c014af8236be09c54c13452e4539c")

    variant("shared", default=True, description="Build shared libraries")

    depends_on("fortran", type="build")  # generated
    depends_on("meson@0.60.0:", type="build", when="build_system=meson")

    depends_on("pkgconfig", type="build")


class CMakeBuilder(cmake.CMakeBuilder):
    def cmake_args(self):
        return [self.define_from_variant("BUILD_SHARED_LIBS", "shared")]


class MesonBuilder(meson.MesonBuilder):
    def meson_args(self):
        return [
            "-Ddefault_library={0}".format("shared" if "+shared" in self.spec else "static"),
        ]
