#!/bin/bash
if [ ! -v MATX_VERSION_TAG ]; then
   MATX_VERSION_TAG="13.0.1_ubuntu24.04"
fi

if [ ! -v MATX_REPO ]; then
   MATX_REPO="ghcr.io/nvidia/matx/"
fi

if [ ! -v MATX_IMAGE_NAME ]; then
   MATX_IMAGE_NAME="release"
fi

if [ ! -v MATX_INSTANCE_NAME ]; then
   MATX_INSTANCE_NAME="c_matx"
fi
    
if [ -z "$MATX_PLATFORM" ]; then
    case "$(arch)" in
        "x86_64")
            MATX_PLATFORM="linux/amd64"
            ;;
        "aarch64")
            MATX_PLATFORM="linux/arm64"
            ;;
        *)
            echo "Unsupported arch type"
            exit 1
            ;;
    esac
fi
