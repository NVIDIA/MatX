# Use the base image
FROM ghcr.io/nvidia/matx/release:latest

ARG REMOTE_USER
ARG REMOTE_UID
ARG REMOTE_GID

RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-numpy && \
    sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Create the user
RUN groupadd --gid $REMOTE_GID $REMOTE_USER \
    && useradd --uid $REMOTE_UID --gid $REMOTE_GID -m $REMOTE_USER \
    #
    # [Optional] Add sudo support. Omit if you don't need sudo.
    && echo $REMOTE_USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$REMOTE_USER \
    && chmod 0440 /etc/sudoers.d/$REMOTE_USER

# Set user
USER $REMOTE_USER

# Copy all files in the current folder to the container
COPY --chown=$REMOTE_USER:$REMOTE_USER . /workspaces/MatX

# Set the working directory
WORKDIR /workspaces/MatX
