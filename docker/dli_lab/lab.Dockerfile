FROM ghcr.io/nvidia/matx/production:latest AS devel



# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  git \
  python3 \
  python3-pip \
  python3-dev \
  python3-venv \
  nodejs \
  npm \
  libpugixml-dev \
  lsb-release \
  wget \
  software-properties-common \
  gnupg \
  && rm -rf /var/lib/apt/lists/*

# Add LLVM repository
RUN apt-get update && apt-get install -y wget && \
  wget https://apt.llvm.org/llvm.sh && \
  chmod +x llvm.sh && \
  ./llvm.sh 19 && \
  ./llvm.sh 18

# Create symbolic link for llvm-config
RUN ln -s /usr/bin/llvm-config-19 /usr/bin/llvm-config

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages
RUN pip3 install --no-cache-dir \
  jupyter \
  jupyterlab \
  cmake \
  nlohmann-json==3.11.2 \
  xtl \
  pugixml

# Install doctest from source
RUN git clone https://github.com/doctest/doctest.git /tmp/doctest \
  && cd /tmp/doctest \
  && cmake -B build -S . \
  && cmake --build build --target install

# Create and set working directory
RUN mkdir /opt/xeus
WORKDIR /opt/xeus

RUN git clone https://github.com/root-project/llvm-project.git && \
  cd llvm-project && \
  git checkout cling-latest && \
  cd .. && \
  git clone https://github.com/root-project/cling.git

RUN  mkdir cling-build && cd cling-build && \
  cmake -DLLVM_EXTERNAL_PROJECTS=cling -DLLVM_EXTERNAL_CLING_SOURCE_DIR=../cling/ -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_TARGETS_TO_BUILD="host;NVPTX" -DCMAKE_BUILD_TYPE=Release ../llvm-project/llvm && \
  cmake --build . --parallel 72 --target cling && \
  cmake --build . --parallel 72 --target clang

WORKDIR /opt/xeus/cling/tools/Jupyter/kernel
ENV PATH="/opt/xeus/cling-build/bin/:${PATH}"
RUN cd /opt/xeus/cling/tools/Jupyter/kernel && git clone https://github.com/NVIDIA/MatX.git && cd MatX && mkdir build && cd build && cmake -DMATX_EN_X86_FFTW=ON -DMATX_EN_OPENBLAS=ON -DMATX_EN_CUTENSOR=ON ..
RUN cd /opt/xeus/cling/tools/Jupyter/kernel && pip install -e . && jupyter-kernelspec install --user cling-cpp17
RUN cd /opt/xeus/cling-build/tools/cling/tools/Jupyter/ && make -j
#RUN cd /opt/xeus/cling/tools/Jupyter/kernel && jupyter-kernelspec install --user cling-cpp17

RUN pip install bash_kernel
RUN python -m bash_kernel.install

# Create jupyter config directory
RUN mkdir -p ~/.jupyter

# Expose Jupyter port
EXPOSE 8888

# Set working directory for Jupyter
WORKDIR /notebooks

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
