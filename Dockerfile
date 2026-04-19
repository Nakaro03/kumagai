FROM jupyter/base-notebook

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
  make \
  curl \
  file \
  git \
  libmecab-dev \
  mecab \
  mecab-ipadic-utf8 \
  build-essential \
  libffi-dev \
  libssl-dev \
  libjpeg-dev \
  libpng-dev \
  libfreetype6-dev \
  pkg-config

# Install MeCab with Neologd
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
RUN mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -y

# Copy requirements and install Python packages
COPY requirements.txt $PWD
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter extensions for better visualization
RUN pip install --no-cache-dir \
  jupyterlab-plotly \
  jupyterlab-widgets \
  ipywidgets

# Set working directory
WORKDIR /home/jovyan/work

# Copy the project files
COPY . /home/jovyan/work/

# Set permissions
RUN chown -R jovyan:users /home/jovyan/work
USER jovyan

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
