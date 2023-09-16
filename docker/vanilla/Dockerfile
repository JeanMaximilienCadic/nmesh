FROM python:3.12.0rc1-slim

# Add system deps
RUN apt update && \
        apt install \
        libgl1-mesa-glx \
        meshlab \
        wget \
        unzip \
        make \
        -y
