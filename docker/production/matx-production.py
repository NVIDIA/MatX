#!/usr/bin/env python

import hpccm
from hpccm.building_blocks import gnu, mlnx_ofed, nvshmem, cmake
from hpccm.primitives import baseimage

DOXYGEN_VER = "1.9.6"
GDRCOPY_HOME = "/usr/local/gdrcopy"
PYBIND11_VER = "2.7.1"
FFTW_VER="3.3.10"
OPENBLAS_VER="0.3.27"
BLIS_VER="1.0"

if cpu_target == 'x86_64':
    TARGETARCH='amd64'
elif cpu_target == 'aarch64':
    TARGETARCH='arm64'
else:
    raise RuntimeError("Unsupported platform")

Stage0 = hpccm.Stage()
Stage0 += baseimage(image='nvidia/cuda:12.6.2-devel-ubuntu22.04', _as='devel', _distro="ubuntu22")

Stage0 += packages(ospackages=[
    'bison',
    'clang-tidy',
    'curl',
    'flex',
    'ghostscript',
    'git',
    'libjs-mathjax',
    'liblapacke-dev',    
    'libopenblas64-openmp-dev',
    'lcov',
    'ninja-build',
    'numactl',
    'python3-pip',
    'python3-dev',
    'sudo',
    'texlive-font-utils',
    'valgrind',
    'vim',
])

#Stage0 += shell(commands=["mv /usr/lib/python3.12/EXTERNALLY-MANAGED /usr/lib/python3.12/EXTERNALLY-MANAGED.old"])
Stage0 += pip(pip="pip3", upgrade=True, packages=[
    'breathe',
    'cupy-cuda12x',
    'hpccm',
    'numpy',
    'pandas',
    'plotly==5.2.1',
    'pybind11',
    'scipy',
    'sphinx',
    'sphinx_book_theme',
    'sphinx-rtd-theme',
])

Stage0 += gnu()
Stage0 += cmake(eula=True, version="3.26.4")
Stage0 += nsight_compute(eula=True)
Stage0 += nsight_systems()

Stage0 += shell(commands=["wget https://doxygen.nl/files/doxygen-{}.src.tar.gz".format(DOXYGEN_VER),
                          "tar -zxf doxygen-{}.src.tar.gz".format(DOXYGEN_VER),
                          "cd doxygen-{} && mkdir build && cd build && cmake .. && make -j && make install".format(DOXYGEN_VER)])
Stage0 += shell(commands=["python3 --version"])

Stage0 += shell(commands=[f"wget https://www.fftw.org/fftw-{FFTW_VER}.tar.gz && tar -xzf fftw-{FFTW_VER}.tar.gz",
                          f"cd fftw-{FFTW_VER} && ./configure --enable-sse2 --enable-avx2 --enable-shared --enable-avx512 --enable-openmp --enable-float && make && make install",
                          "./configure --enable-sse2 --enable-avx2 --enable-avx512 --enable-openmp && make && sudo make install"])

# Stage0 += shell(commands=[f"cd /tmp && wget https://github.com/OpenMathLib/OpenBLAS/releases/download/v{OPENBLAS_VER}/OpenBLAS-{OPENBLAS_VER}.tar.gz && tar -zxvf OpenBLAS-{OPENBLAS_VER}.tar.gz && cd OpenBLAS-{OPENBLAS_VER}",
#                          "make -j && sudo make USE_OPENMP=1 INTERFACE64=1 install"])

Stage0 += shell(commands=["curl -L https://coveralls.io/coveralls-linux.tar.gz | tar -xz -C /usr/local/bin"])

Stage0 += shell(commands=[f"cd /tmp && wget https://github.com/flame/blis/archive/refs/tags/{BLIS_VER}.tar.gz && tar -zxvf {BLIS_VER}.tar.gz && cd blis-{BLIS_VER}",
                         "./configure --enable-threading=openmp --enable-cblas -b 64 auto && sudo make -j install"])


# # Install fixuid
# Stage0 += shell(commands=[
#     'addgroup --gid 1000 matx',
#     'adduser --uid 1000 --ingroup matx --home /home/matx --shell /bin/sh --disabled-password --gecos "" matx',
#     'USER=matx',
#     'GROUP=matx',
#     f'curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.5/fixuid-0.5-linux-{TARGETARCH}.tar.gz | tar -C /usr/local/bin -xzf -',
#     'chown root:root /usr/local/bin/fixuid',
#     'chmod 4755 /usr/local/bin/fixuid',
#     'mkdir -p /etc/fixuid',
#     'printf "user: $USER\\ngroup: $GROUP\\n" > /etc/fixuid/config.yml',
#     '/bin/echo "matx ALL = (root) NOPASSWD: ALL" >> /etc/sudoers',
#    ])

print(Stage0)
