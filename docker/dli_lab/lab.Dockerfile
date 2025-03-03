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

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN mkdir /root/.ipython/profile_default
RUN mkdir /root/.ipython/extensions
RUN echo c.InteractiveShellApp.extensions = ['run_matx'] >> /root/.ipython/profile_default/ipython_config.py
RUN cp ./run_matx.py /root/.ipython/extensions/

# Install Python packages
RUN pip3 install --no-cache-dir \
  jupyter \
  jupyterlab \
  cmake \
  nlohmann-json==3.11.2 \
  xtl \
  pugixml

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
