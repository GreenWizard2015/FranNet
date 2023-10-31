FROM python:3.10

# Some stuff to make it work on Hugging Face
RUN apt-get update && apt-get install -y libsm6 libxext6 cmake libgl1-mesa-glx
RUN --mount=target=pre-requirements.txt,source=pre-requirements.txt 	pip install --no-cache-dir -r pre-requirements.txt
RUN pip install --no-cache-dir gradio==3.40.0 tensorflow tensorflow-probability opencv-python wandb imageio[ffmpeg]

# Set up environment
COPY --link --chown=1000 ./ /home/user/app
WORKDIR /home/user/app
RUN useradd -m -u 1000 user && cd /home/user/app
USER user
ENV PATH=/home/user/.local/bin:$PATH

# Clone the repo and install the requirements
ADD "https://api.github.com/repos/GreenWizard2015/FranNet/git/ref/heads/main" skipcache
RUN git clone https://github.com/GreenWizard2015/FranNet/ && \
    cp -r FranNet/huggingface/* . && rm -rf FranNet/huggingface && \
    cp -r FranNet/* . && \
    rm -rf FranNet && \
    pip install --no-cache-dir -r requirements.txt
    
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]