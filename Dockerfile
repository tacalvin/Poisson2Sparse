FROM continuumio/anaconda3

# Install system dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        curl \
        git \
    && apt-get clean

# Install python miniconda3 + requirements
ENV MINICONDA_HOME="/opt/miniconda"
ENV PATH="${MINICONDA_HOME}/bin:${PATH}"
COPY environment.yaml environment.yaml
RUN conda env create --name p2s --file=environment.yaml

# 
WORKDIR /Poisson2Sparse
COPY src /Poisson2Sparse

ENTRYPOINT [ "bash"]
