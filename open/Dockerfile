FROM continuumio/miniconda3:24.5.0-0

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3 \
    libosmesa6 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

SHELL ["conda", "run", "-n", "d4rl-release", "/bin/bash", "-c"]

CMD ["bash"]
