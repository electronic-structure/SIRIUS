# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.build_systems import cmake, meson
from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.meson import MesonPackage

from spack.package import *


class MctcLib(MesonPackage, CMakePackage):
    """Modular computation toolchain library for quantum chemistry file IO"""

    homepage = "https://github.com/grimme-lab/mctc-lib"
    url = "https://github.com/grimme-lab/mctc-lib/releases/download/v0.0.0/mctc-lib-0.0.0.tar.xz"
    git = "https://github.com/grimme-lab/mctc-lib"

    maintainers("awvwgk")

    license("Apache-2.0")

    build_system("cmake", "meson", default="meson")

    version("main", branch="main")

    version("0.5.1", sha256="a93ea3e50a1950745df01601bfd672d485f0367660f7076dbe73e422e7d4e2ac")
    version("0.5.0", sha256="afd0dd4e40c3441432f077e14112962273ccc25abb00db05d7559fec3b0f1505")
    version("0.4.2", sha256="ce1e962c79d871d3705be590aef44f07ca296843b85e164307290f8324769406")
    version("0.4.1", sha256="57fe4610c4fa21d0b797f88b68481c7be1e7d291daa12063caed51bee779b88a")
    version("0.4.0", sha256="43f6988fc5a2c8d2d8397c6ef8d55f745c9869dd94a7dc9c099a0e1b7f423e40")
    version("0.3.2", sha256="8c4ebdf9d81272f0dfa0bfa6c7fecd51f1f3d83d3629c719298d9f349de6ee0b")
    version("0.3.1", sha256="a5032a0bbbbacc952037c5215b71aa6b438767a84bafb60fda25ba43c8835513")
    version("0.3.0", sha256="81f3edbf322e6e28e621730a796278498b84af0f221f785c537a315312059bf0")

    variant("json", default=False, description="Enable support for JSON")
    variant("openmp", default=False, description="Enable OpenMP support")

    depends_on("fortran", type="build")  # generated

    depends_on("meson@0.57.2:", type="build", when="build_system=meson")
    depends_on("json-fortran@8:", when="@:0.4.2+json")
    depends_on("pkgconfig", type="build")

    for build_system in ["cmake", "meson"]:
        depends_on(f"jonquil build_system={build_system}", when=f"build_system={build_system}")
        depends_on(
            f"toml-f build_system={build_system}", when=f"@0.4.2:+json build_system={build_system}"
        )


class CMakeBuilder(cmake.CMakeBuilder):
    def cmake_args(self):
        return [
            self.define_from_variant("WITH_JSON", "json"),
            self.define_from_variant("WITH_OpenMP", "openmp"),
            "-DBUILD_SHARED_LIBS=On"
        ]


class MesonBuilder(meson.MesonBuilder):
    def meson_args(self):
        return [
            "-Djson={0}".format("enabled" if "+json" in self.spec else "disabled"),
            "-Dopenmp={0}".format("true" if "+openmp" in self.spec else "false"),
        ]
