FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ARG workspace_path

ENV DEBIAN_FRONTEND=noninteractive

###### Basic installation for docker development ######
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

###### CRITICAL: Pin NumPy to 1.x version first ######
# Install and pin NumPy to prevent any upgrades to 2.x
RUN pip3 install --no-cache-dir numpy==1.24.3 && \
    pip3 install --no-cache-dir --no-deps numpy==1.24.3

###### Install OpenCV with compatible NumPy ######
# Install opencv-python-headless to avoid GUI dependencies and ensure compatibility
RUN pip3 install --no-cache-dir opencv-python-headless==4.8.0.74

###### Install MMCV with CUDA support ######
# This specific version is pre-built for PyTorch 2.0.1 + CUDA 11.7
# Using cu117 to match the base image CUDA version
RUN pip3 install --no-cache-dir mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html

###### Core scientific packages with NumPy constraint ######
RUN pip3 install --no-cache-dir --default-timeout=100 \
    "scikit-learn<1.4.0" \
    "pandas<2.0.0" \
    "scipy<1.12.0" \
    "matplotlib<3.8.0"

###### Computer vision and deep learning packages ######
RUN pip3 install --no-cache-dir --default-timeout=100 \
    "Pillow<10.0.0" \
    "scikit-image<0.22.0" \
    imageio-ffmpeg

###### PyTorch utilities and monitoring ######
RUN pip3 install --no-cache-dir --default-timeout=100 \
    torchsummary \
    torchsummaryX \
    "tensorboard<2.15.0" \
    "tensorboardX<2.7.0" \
    torch-tb-profiler \
    flopco-pytorch

###### Additional packages for PGSPN ######
RUN pip3 install --no-cache-dir --default-timeout=100 \
    "timm<0.9.0" \
    configargparse \
    stories \
    tqdm \
    pyyaml

###### Final check to ensure NumPy 1.x is maintained ######
# Reinstall NumPy 1.x if anything upgraded it
RUN pip3 install --no-cache-dir --force-reinstall numpy==1.24.3


###### Environment variables for GPU optimization ######
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512


# Default command
CMD ["/bin/bash"]