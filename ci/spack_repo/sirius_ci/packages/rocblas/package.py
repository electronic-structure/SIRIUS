# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import re

from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage

from spack.package import *


class Rocblas(CMakePackage):
    """Radeon Open Compute BLAS library"""

    homepage = "https://github.com/ROCm/rocBLAS/"
    git = "https://github.com/ROCm/rocm-libraries.git"

    tags = ["rocm"]
    maintainers("cgmb", "srekolam", "renjithravindrankannath", "haampie", "afzpatel")
    libraries = ["librocblas"]
    license("MIT")

    def url_for_version(self, version):
        if version <= Version("7.1.1"):
            url = "https://github.com/ROCm/rocBLAS/archive/refs/tags/rocm-{0}.tar.gz"
        else:
            url = "https://github.com/ROCm/rocm-libraries/archive/rocm-{0}.tar.gz"
        return url.format(version)

    version("7.2.0", sha256="8ad5f4a11f1ed8a7b927f2e65f24083ca6ce902a42021a66a815190a91ccb654")
    version("7.1.1", sha256="29d43270ccf5d4818d261993f964d4fce4bd0a55c2b6dde60d1529b6c227a873")
    version("7.1.0", sha256="54f38222d0e58344cf5c86f151d418c071b59145297fd2ed953bb561df1e12c3")
    version("7.0.2", sha256="8398cda68242db2386abc9eaf00c3588bb27e2b382e29be2bc5624c2d4ac8a99")
    version("7.0.0", sha256="337a77cec31927e484672002d245d3aebf7a67e95658a8477fc593c95cf281fb")
    version("6.4.3", sha256="754dcc88b30468a2293d2406d7fe40f78dc92dd77c193758f937532217ecdad3")
    version("6.4.2", sha256="703226c458bb3dd1155aad8bdc02cdae2ff789c6b44e41e4a49ae28e40baff98")
    version("6.4.1", sha256="517950ff6b3715dee8b2bcfbdd3968c65e1910e4b8e353e148574ae08aa6dc73")
    version("6.4.0", sha256="ab8e75c9f98d17817a650aa4f06ff1e6c6af92cd143079e361cb6a0c96676aaa")
    version("6.3.3", sha256="73e91bd50c920b818742fa5bf9990c0676be5bfbafe321d5781607dc2ce27060")
    version("6.3.2", sha256="455cad760d926c21101594197c4456f617e5873a8f17bb3e14bd762018545a9e")
    version("6.3.1", sha256="88d2de6ce6b23a157eea8be63408350848935e4dfc3e27e5f2add78834c6d6ba")
    version("6.3.0", sha256="051f53bb69a9aba55a0c66c32688bf6af80e29e4a6b56b380b3c427e7a6aff9d")
    version("6.2.4", sha256="8bacf74e3499c445f1bb0a8048df1ef3ce6f72388739b1823b5784fd1e8aa22a")
    version("6.2.1", sha256="cf3bd7b47694f95f387803191615e2ff5c1106175473be7a5b2e8eb6fb99179f")
    version("6.2.0", sha256="184e9b39dcbed57c25f351b047d44c613f8a2bbab3314a20c335f024a12ad4e5")
    version("6.1.2", sha256="1e83918bd7b28ec9ee292c6fb7eb0fc5f4db2d5d831a9a3db541f14a90c20a1a")
    version("6.1.1", sha256="c920742fb8f45512c360cdb40e37d0ac767f042e52f1981264853dab5ec2c876")
    version("6.1.0", sha256="af00357909da60d82618038aa9a3cc1f9d4ce1bdfb54db20ec746b592d478edf")
    version("6.0.2", sha256="d1bf31063a2d349797b88c994c91d05f94e681bafb5550ad9b53529703d89dbb")
    version("6.0.0", sha256="befa4a75f1de0ea37f2358d4c2de5406d7bce671ca9936e2294b64d3b3bafb60")
    version("5.7.1", sha256="2984a5ed0ea5a05d40996ee3fddecb24399cbe8ea3e4921fc254e54d8f52fe4f")
    version("5.7.0", sha256="024edd98de9687ee5394badc4dd4c543eef4eb3f71c96ff64100705d851e1744")

    amdgpu_targets = ROCmPackage.amdgpu_targets

    variant(
        "amdgpu_target",
        description="AMD GPU architecture",
        values=auto_or_any_combination_of(*amdgpu_targets),
        sticky=True,
    )
    variant("tensile", default=True, description="Use Tensile as a backend")
    variant("asan", default=False, description="Build with address-sanitizer enabled or disabled")
    variant("hipblaslt", default=True, when="@6.3:", description="Build with hipblaslt")

    conflicts("+asan", when="os=rhel9")
    conflicts("+asan", when="os=centos7")
    conflicts("+asan", when="os=centos8")

    depends_on("c", type="build")  # generated
    depends_on("cxx", type="build")  # generated
    depends_on("fortran", type="build")  # generated

    depends_on("cmake@3.16.8:", type="build")

    depends_on("googletest@1.10.0:", type="test")
    depends_on("amdblis", type="test")

    for ver in [
        "6.2.0",
        "6.2.1",
        "6.2.4",
        "6.3.0",
        "6.3.1",
        "6.3.2",
        "6.3.3",
        "6.4.0",
        "6.4.1",
        "6.4.2",
        "6.4.3",
        "7.0.0",
        "7.0.2",
        "7.1.0",
        "7.1.1",
        "7.2.0",
    ]:
        depends_on(f"rocm-smi-lib@{ver}", type="test", when=f"@{ver}")

    for ver in [
        "5.7.0",
        "5.7.1",
        "6.0.0",
        "6.0.2",
        "6.1.0",
        "6.1.1",
        "6.1.2",
        "6.2.0",
        "6.2.1",
        "6.2.4",
        "6.3.0",
        "6.3.1",
        "6.3.2",
        "6.3.3",
        "6.4.0",
        "6.4.1",
        "6.4.2",
        "6.4.3",
        "7.0.0",
        "7.0.2",
        "7.1.0",
        "7.1.1",
    ]:
        depends_on(f"rocm-openmp-extras@{ver}", type="test", when=f"@{ver}")

    for ver in [
        "5.7.0",
        "5.7.1",
        "6.0.0",
        "6.0.2",
        "6.1.0",
        "6.1.1",
        "6.1.2",
        "6.2.0",
        "6.2.1",
        "6.2.4",
        "6.3.0",
        "6.3.1",
        "6.3.2",
        "6.3.3",
        "6.4.0",
        "6.4.1",
        "6.4.2",
        "6.4.3",
        "7.0.0",
        "7.0.2",
        "7.1.0",
        "7.1.1",
        "7.2.0",
    ]:
        depends_on(f"hip@{ver}", when=f"@{ver}")
        depends_on(f"llvm-amdgpu@{ver}", type="build", when=f"@{ver}")
        depends_on(f"rocminfo@{ver}", type="build", when=f"@{ver}")
        depends_on(f"rocm-cmake@{ver}", type="build", when=f"@{ver}")

    for ver in [
        "6.3.0",
        "6.3.1",
        "6.3.2",
        "6.3.3",
        "6.4.0",
        "6.4.1",
        "6.4.2",
        "6.4.3",
        "7.0.0",
        "7.0.2",
        "7.1.0",
        "7.1.1",
        "7.2.0",
    ]:
        depends_on(f"hipblaslt@{ver}", when=f"@{ver} +hipblaslt")

    for ver in ["6.4.0", "6.4.1", "6.4.2", "6.4.3", "7.0.0", "7.0.2", "7.1.0", "7.1.1", "7.2.0"]:
        depends_on(f"roctracer-dev@{ver}", when=f"@{ver}")

    depends_on("python@3.6:", type="build")

    with when("+tensile"):
        depends_on("msgpack-c@3:")

        depends_on("py-virtualenv", type="build")
        depends_on("perl-file-which", type="build")
        depends_on("py-pyyaml", type="build")
        depends_on("py-wheel", type="build")
        depends_on("py-msgpack", type="build")
        depends_on("py-pip", type="build")
        depends_on("py-joblib", type="build")
        depends_on("procps", type="build")

    for t_version, t_commit in [
        ("@5.7.0", "97e0cfc2c8cb87a1e38901d99c39090dc4181652"),
        ("@5.7.1", "97e0cfc2c8cb87a1e38901d99c39090dc4181652"),
        ("@6.0.0", "17df881bde80fc20f997dfb290f4bb4b0e05a7e9"),
        ("@6.0.2", "17df881bde80fc20f997dfb290f4bb4b0e05a7e9"),
        ("@6.1.0", "2b55ccf58712f67b3df0ca53b0445f094fcb96b2"),
        ("@6.1.1", "2b55ccf58712f67b3df0ca53b0445f094fcb96b2"),
        ("@6.1.2", "2b55ccf58712f67b3df0ca53b0445f094fcb96b2"),
        ("@6.2.0", "dbc2062dced66e4cbee8e0591d76e0a1588a4c70"),
        ("@6.2.1", "dbc2062dced66e4cbee8e0591d76e0a1588a4c70"),
        ("@6.2.4", "81ae9537671627fe541332c0a5d953bfd6af71d6"),
        ("@6.3.0", "aca95d1743c243dd0dd0c8b924608bc915ce1ae7"),
        ("@6.3.1", "aca95d1743c243dd0dd0c8b924608bc915ce1ae7"),
        ("@6.3.2", "aca95d1743c243dd0dd0c8b924608bc915ce1ae7"),
        ("@6.3.3", "aca95d1743c243dd0dd0c8b924608bc915ce1ae7"),
        ("@6.4.0", "be49885fce2a61b600ae4593f1c2d00c8b4fa11e"),
        ("@6.4.1", "be49885fce2a61b600ae4593f1c2d00c8b4fa11e"),
        ("@6.4.2", "be49885fce2a61b600ae4593f1c2d00c8b4fa11e"),
        ("@6.4.3", "be49885fce2a61b600ae4593f1c2d00c8b4fa11e"),
        ("@7.0.0", "cca3c8136aa812109629e6291ce9f0ca846b68d3"),
        ("@7.0.2", "63c27e505cb532ff8e568d737bfdbd9e1d024665"),
        ("@7.1.0", "0c8314da90fee8cf3b16dcb1bbc75bc1266e123f"),
        ("@7.1.1", "0c8314da90fee8cf3b16dcb1bbc75bc1266e123f"),
    ]:
        resource(
            name="Tensile",
            git="https://github.com/ROCm/Tensile.git",
            commit=t_commit,
            placement="Tensile",
            when=f"{t_version} +tensile",
        )

    patch("0007-add-rocm-openmp-extras-include-dir.patch", when="@5.7")
    patch("0008-link-roctracer.patch", when="@6.4")
    patch("0009-use-rocm-smi-config.patch", when="@6.4:7.1")
    patch("0001-remove-blas-override.patch", when="@7.1")
    patch("0001-remove-blas-override-7.2.patch", when="@7.2:")
    patch(
        "https://github.com/ROCm/rocm-libraries/commit/b3b20a3ea53051a14f30c28e577620c0beeea57c.patch?full_index=1",
        sha256="1f436c5ad03c8fdc021f309a1ad84d4356f30c39c4cc940bb8267841561bf5f1",
        when="@7.2",
    )

    @property
    def root_cmakelists_dir(self):
        if self.spec.satisfies("@7.2:"):
            return "projects/rocblas"
        else:
            return "."

    def setup_build_environment(self, env: EnvironmentModifications) -> None:
        env.set("CXX", self.spec["hip"].hipcc)
        if self.spec.satisfies("+asan"):
            env.set("CC", f"{self.spec['llvm-amdgpu'].prefix}/bin/clang")
            env.set("CXX", f"{self.spec['llvm-amdgpu'].prefix}/bin/clang++")
            env.set("ASAN_OPTIONS", "detect_leaks=0")
            env.set("CFLAGS", "-fsanitize=address -shared-libasan")
            env.set("CXXFLAGS", "-fsanitize=address -shared-libasan")
            env.set("LDFLAGS", "-fuse-ld=lld")

    @classmethod
    def determine_version(cls, lib):
        match = re.search(r"lib\S*\.so\.\d+\.\d+\.(\d)(\d\d)(\d\d)", lib)
        if match:
            ver = "{0}.{1}.{2}".format(
                int(match.group(1)), int(match.group(2)), int(match.group(3))
            )
        else:
            ver = None
        return ver

    def cmake_args(self):
        args = [
            self.define("BUILD_CLIENTS_TESTS", self.run_tests),
            self.define("BUILD_CLIENTS_BENCHMARKS", "OFF"),
            self.define("BUILD_CLIENTS_SAMPLES", "OFF"),
            self.define("RUN_HEADER_TESTING", "OFF"),
            self.define_from_variant("BUILD_WITH_TENSILE", "tensile"),
            self.define("CMAKE_INSTALL_LIBDIR", "lib"),
            self.define("Tensile_CODE_OBJECT_VERSION", "default"),
        ]
        if self.run_tests:
            args.append(self.define("LINK_BLIS", "ON"))
            args.append(
                self.define("BLIS_INCLUDE_DIR", self.spec["amdblis"].prefix + "/include/blis/")
            )
            args.append(
                self.define("BLAS_LIBRARY", self.spec["amdblis"].prefix + "/lib/libblis.a")
            )
            if self.spec.satisfies("@:7.1"):
                args.append(
                    self.define("ROCM_OPENMP_EXTRAS_DIR", self.spec["rocm-openmp-extras"].prefix)
                )

        if "+tensile" in self.spec:
            if self.spec.satisfies("@:6.2"):
                tensile_compiler = "hipcc"
            else:
                tensile_compiler = f"{self.spec['llvm-amdgpu'].prefix}/bin/amdclang++"
            args += [
                self.define("Tensile_COMPILER", tensile_compiler),
                self.define("Tensile_LOGIC", "asm_full"),
                self.define("BUILD_WITH_TENSILE_HOST", "@3.7.0:" in self.spec),
                self.define("Tensile_LIBRARY_FORMAT", "msgpack"),
            ]
            if self.spec.satisfies("@:7.1"):
                tensile_path = join_path(self.stage.source_path, "Tensile")
                args.append(self.define("Tensile_TEST_LOCAL_PATH", tensile_path))
            # Restrict the number of jobs Tensile can spawn.
            # If we don't specify otherwise, Tensile creates a job per available core,
            # and that consumes a lot of system memory.
            # https://github.com/ROCm/Tensile/blob/93e10678a0ced7843d9332b80bc17ebf9a166e8e/Tensile/Parallel.py#L38
            args.append(self.define("Tensile_CPU_THREADS", min(16, make_jobs)))

        if "auto" not in self.spec.variants["amdgpu_target"]:
            if self.spec.satisfies("@7.1:"):
                args.append(self.define_from_variant("GPU_TARGETS", "amdgpu_target"))
            else:
                args.append(self.define_from_variant("AMDGPU_TARGETS", "amdgpu_target"))

        # See https://github.com/ROCm/rocBLAS/issues/1196
        if self.spec.satisfies("^cmake@3.21.0:3.21.2"):
            args.append(self.define("__skip_rocmclang", "ON"))

        if self.spec.satisfies("@:6.3.1"):
            args.append(self.define("BUILD_FILE_REORG_BACKWARD_COMPATIBILITY", True))
        if self.spec.satisfies("@6.3:"):
            args.append(self.define_from_variant("BUILD_WITH_HIPBLASLT", "hipblaslt"))
        if self.spec.satisfies("@7.1:"):
            args.append(self.define("ROCTX_PATH", self.spec["roctracer-dev"].prefix))
        return args

    @run_after("build")
    @on_package_attributes(run_tests=True)
    def check_build(self):
        exe = Executable(join_path(self.build_directory, "clients", "staging", "rocblas-test"))
        exe("--gtest_filter=*quick*-*known_bug*")
