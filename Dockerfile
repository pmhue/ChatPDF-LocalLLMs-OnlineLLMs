# Use PyTorch with CUDA as base image
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Install required dependencies
RUN apt-get update && apt-get install -y \
    curl \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Ensure ollama is in PATH
ENV PATH="/root/.ollama/bin:$PATH"

# Create working directory
RUN mkdir /chatpdf
WORKDIR /chatpdf/src

# Copy project files to working dirs
COPY . /chatpdf/src/

# Install requirement packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the ports for Ollama and Streamlit
EXPOSE 11434 8501

# Start Ollama in the background and run Streamlit
CMD ollama serve & \
    while ! curl -s http://localhost:11434/api/info > /dev/null; do sleep 1; done && \
    # ollama pull mxbai-embed-large && \
    # ollama pull deepseek-r1:latest && \
    streamlit run app.py