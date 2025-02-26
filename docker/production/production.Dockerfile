FROM nvidia/cuda:12.6.2-devel-ubuntu22.04 AS devel

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bison \
        clang-tidy \
        curl \
        flex \
        ghostscript \
        git \
        lcov \
        libjs-mathjax \
        liblapacke-dev \
        libopenblas64-openmp-dev \
        ninja-build \
        numactl \
        python3-dev \
        python3-pip \
        sudo \
        texlive-font-utils \
        valgrind \
        vim && \
    rm -rf /var/lib/apt/lists/*

# pip
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        python3-wheel && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 --no-cache-dir install --upgrade pip && \
    pip3 --no-cache-dir install breathe cupy-cuda12x hpccm numpy pandas plotly==5.2.1 pybind11 scipy sphinx sphinx_book_theme sphinx-rtd-theme

# GNU compiler
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        g++ \
        gcc \
        gfortran && \
    rm -rf /var/lib/apt/lists/*

# CMake version 3.26.4
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        make \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh && \
    mkdir -p /usr/local && \
    /bin/sh /var/tmp/cmake-3.26.4-linux-x86_64.sh --prefix=/usr/local --skip-license && \
    rm -rf /var/tmp/cmake-3.26.4-linux-x86_64.sh
ENV PATH=/usr/local/bin:$PATH

# NVIDIA Nsight Compute 2022.4.0
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /usr/share/keyrings && \
    rm -f /usr/share/keyrings/nvidia.gpg && \
    wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub | gpg --dearmor -o /usr/share/keyrings/nvidia.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /" >> /etc/apt/sources.list.d/hpccm.list && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        nsight-compute-2022.4.0 && \
    rm -rf /var/lib/apt/lists/*
ENV NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT=1 \
    PATH=/opt/nvidia/nsight-compute/2022.4.0:$PATH

# NVIDIA Nsight Systems 2022.5.1
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /usr/share/keyrings && \
    rm -f /usr/share/keyrings/nvidia.gpg && \
    wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub | gpg --dearmor -o /usr/share/keyrings/nvidia.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /" >> /etc/apt/sources.list.d/hpccm.list && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        nsight-systems-cli-2022.5.1 && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://doxygen.nl/files/doxygen-1.9.6.src.tar.gz && \
    tar -zxf doxygen-1.9.6.src.tar.gz && \
    cd doxygen-1.9.6 && mkdir build && cd build && cmake .. && make -j && make install

RUN python3 --version

RUN wget https://www.fftw.org/fftw-3.3.10.tar.gz && tar -xzf fftw-3.3.10.tar.gz && \
    cd fftw-3.3.10 && ./configure --enable-sse2 --enable-avx2 --enable-shared --enable-avx512 --enable-openmp --enable-float && make && make install && \
    ./configure --enable-sse2 --enable-avx2 --enable-avx512 --enable-openmp && make && sudo make install

RUN curl -L https://coveralls.io/coveralls-linux.tar.gz | tar -xz -C /usr/local/bin

RUN cd /tmp && wget https://github.com/flame/blis/archive/refs/tags/1.0.tar.gz && tar -zxvf 1.0.tar.gz && cd blis-1.0 && \
    ./configure --enable-threading=openmp --enable-cblas -b 64 auto && sudo make -j install



