import setuptools

setuptools.setup(
    name="sirius",
    version="0.5",
    author="",
    author_email="simon.pintarelli@cscs.ch",
    description="pySIRIUS",
    url="https://github.com/electronic_structure/SIRIUS",
    packages=['sirius'],
    install_requires=['mpi4py', 'voluptuous', 'numpy', 'h5py', 'scipy', 'PyYAML'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7'
)
