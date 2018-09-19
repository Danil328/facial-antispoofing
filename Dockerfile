FROM nvidia/cuda:9.0-base-ubuntu16.04
MAINTAINER danil328

# Install Cuda requirements, basic CLI tools etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        curl \
        git-core \
        iputils-ping \
        libcudnn7=7.0.5.15-1+cuda9.0 \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        wget

# Install Python 3.6
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.6 python3.6-dev

# Link Python to Python 3.6
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1

# Install PIP
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install Python packages
RUN pip --no-cache-dir install \
        colorlover \
        h5py \
        keras \
        matplotlib \
        numpy \
        pandas \
        scikit-image \
        scipy \
        sklearn \
	gdown

RUN pip --no-cache-dir install --upgrade python-dateutil==2.6.1

# Install Python 3.6 extra packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.6-tk

# Install Tensorflow GPU
RUN pip --no-cache-dir install tensorflow-gpu

# Fix pandas excel issue
RUN pip install xlrd openpyxl

# Clean up commands
RUN rm -rf /root/.cache/pip/* && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /usr/local/src/*

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Environment Variables
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

COPY ./ /root
VOLUME /output
VOLUME /test

CMD cd /root && python my_main.py

