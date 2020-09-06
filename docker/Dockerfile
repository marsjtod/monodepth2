#docker build =>
#sudo docker build --tag monodepth2:0.1 ./
#docker run ==> 
#docker run -it --gpus all -v "/mnt/e/workspace":/home/mars/workspace -w "/home/mars/workspace" -p 8888:8888 monodepth2:0.1

ARG UBUNTU_VERSION=18.04
ARG ARCH=
ARG CUDA=10.0
#ARG CUDA=9.0

FROM continuumio/miniconda3:latest AS miniconda
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base

# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=7.6.2.24-1

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

ENV PROJECT=monodepth2

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        cuda-cublas-${CUDA/./-} \
        cuda-cufft-${CUDA/./-} \
        cuda-curand-${CUDA/./-} \
        cuda-cusolver-${CUDA/./-} \
        cuda-cusparse-${CUDA/./-} \
        curl \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

RUN [ ${ARCH} = ppc64le ] || (apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*)

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

ARG USE_PYTHON_3_NOT_2
#ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG _PY_SUFFIX=3
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 

# =============================================================================
# install Anaconda
# =============================================================================

## Updating Ubuntu packages
#RUN apt-get update && yes|apt-get upgrade
RUN apt-get install -y nano
RUN apt-get install -y wget bzip2
RUN apt-get install -y sudo

# Miniconda: get necessary files from build
COPY --from=miniconda /opt/conda /opt/conda

# RUN chmod a+rwx /root
# RUN sudo chown -R root: /opt/conda

# Add user ubuntu with no password, add to sudo group
RUN adduser --disabled-password --gecos '' mars
RUN adduser mars sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER mars
WORKDIR /home/mars/
RUN chmod a+rwx /home/mars/
#RUN echo `pwd`

## ====================================
## miniconda install and setup
ENV SETUSER mars
# Set correct permissions
RUN sudo chown -R $SETUSER: /opt/conda


#   New terminals will have conda active
# If nvidia's Docker image has no .bashrc
# COPY --from=miniconda /home/$SETUSER/.bashrc
# else
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate monodepth2" >> ~/.bashrc

# switch shell sh (default in Linux) to bash
SHELL ["/bin/bash", "-c"]

# give bash access to Anaconda, then normal anaconda commands, e.g. (-q: quiet, -y: answer yes)
## ========================================================================
## !!! "RUN conda create ~~" cause the error, which "can not find conda~~"
## ========================================================================

# RUN conda create -q --name monodepth2 \
#  && conda activate monodepth2
# # && conda install -y your_package
# ## ========================================

# ## =================================
# ## anaconda installation and setup  => working!!!!
# ## ================================
# RUN wget -nv --show-progress --progress=bar:force:noscroll https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh && \
#     bash Anaconda3-2020.07-Linux-x86_64.sh -b
# RUN echo "export PATH="/home/mars/anaconda3/bin:$PATH"" >> ~/.bashrc && \
#     /bin/bash -c "source ~/.bashrc"
# ENV PATH /home/mars/anaconda3/bin:$PATH
# RUN conda update --all

# #==============================================================
# # monodepth2 conda env setup
# #==============================================================
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda update -n base -c defaults conda

RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda create -n monodepth2 python=3.6.6 \
    && conda activate monodepth2 \
    && conda install pytorch=1.0.1 torchvision=0.2.2 -c pytorch

RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda install opencv

RUN source /opt/conda/etc/profile.d/conda.sh \
    && pip install jupyter \
    && pip install tensorboardX==1.4 \
    && pip install matplotlib \
    && pip install scikit-image \
    && pip install IPython 

# Expose Port for jupyter (8881)
# Jtod
EXPOSE 8881

# create project workspace dir
RUN sudo mkdir -p /workspace/experiments
RUN sudo mkdir -p /workspace/${PROJECT}
WORKDIR /workspace/${PROJECT}

# Copy project source last (to avoid cache busting)
WORKDIR /workspace/${PROJECT}
COPY . /workspace/${PROJECT}
ENV PYTHONPATH="/workspace/${PROJECT}:$PYTHONPATH"