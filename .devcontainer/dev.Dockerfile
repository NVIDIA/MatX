# Use the base image
FROM ghcr.io/nvidia/matx/production:v0.9.0_ubuntu22.04-amd64

ARG REMOTE_USER
ARG REMOTE_UID
ARG REMOTE_GID

# Create the user
RUN groupadd --gid $REMOTE_GID $REMOTE_USER \
    && useradd --uid $REMOTE_UID --gid $REMOTE_GID -m $REMOTE_USER \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $REMOTE_USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$REMOTE_USER \
    && chmod 0440 /etc/sudoers.d/$REMOTE_USER

# Set user
USER $REMOTE_USER

# Copy all files in the current folder to the container
COPY --chown=$REMOTE_USER:$REMOTE_USER . /workspaces/radar-simulation

# Set the working directory
WORKDIR /workspaces/radar-simulation