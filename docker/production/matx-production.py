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
    'apt-transport-https',
    'bison',
    'ca-certificates',
    'clang-tidy',
    'curl',
    'flex',
    'ghostscript',
    'git',
    'gnupg',
    'libjs-mathjax',
    'liblapacke-dev',    
    'libopenblas64-openmp-dev',
    'lcov',
    'ninja-build',
    'numactl',
    'python3-dev',
    'python3-pip',
    'python3-setuptools',
    'python3-venv',
    'python3-wheel',
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

Stage0 += shell(commands=["python3 -c \"import cupy, numpy, scipy, pybind11\""])

Stage0 += gnu()
Stage0 += cmake(eula=True, version="3.26.4")
Stage0 += packages(
    apt_keys=[f"https://developer.download.nvidia.com/devtools/repos/ubuntu2204/{TARGETARCH}/nvidia.pub"],
    apt_repositories=[f"deb [signed-by=/usr/share/keyrings/nvidia.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/{TARGETARCH}/ /"],
    force_add_repo=True,
    ospackages=[
        'nsight-compute',
        'nsight-systems',
        'nsight-systems-cli',
    ],
    _apt_key=False)
Stage0 += shell(commands=[
    'ln -sfn "$(find /opt/nvidia/nsight-compute -mindepth 1 -maxdepth 1 -type d -name \'[0-9]*\' | sort -V | tail -n 1)" /opt/nvidia/nsight-compute/latest'
])
Stage0 += environment(variables={
    'NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT': '1',
    'PATH': '/opt/nvidia/nsight-compute/latest:$PATH',
})

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
