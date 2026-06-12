# LeHome Challenge submission: router + ACT specialists + stabilizer
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    linux-libc-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps. Match the local training/eval environment used for the packaged
# checkpoints. Pin image/video packages explicitly because lerobot depends on
# imageio[ffmpeg], which can otherwise resolve inconsistently in a clean Docker
# build.
RUN printf '%s\n' \
        'numpy==1.26.0' \
        'imageio==2.37.0' \
        'imageio-ffmpeg==0.6.0' \
        'opencv-python-headless==4.11.0.86' \
        > /tmp/submission-constraints.txt && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        numpy==1.26.0 \
        torch==2.7.0 \
        torchvision==0.22.0 \
        imageio==2.37.0 \
        imageio-ffmpeg==0.6.0 \
        opencv-python-headless==4.11.0.86 \
        einops && \
    pip install --no-cache-dir --constraint /tmp/submission-constraints.txt \
        datasets==4.1.1 \
        diffusers==0.35.2 \
        "huggingface-hub[cli,hf-transfer]==0.35.3" \
        accelerate==1.12.0 \
        cmake==4.1.3 \
        av==15.1.0 \
        jsonlines==4.0.0 \
        packaging==25.0 \
        pynput==1.8.1 \
        pyserial==3.5 \
        wandb==0.24.0 \
        torchcodec==0.5 \
        draccus==0.10.0 \
        gymnasium==1.2.1 \
        deepdiff==8.6.1 \
        termcolor==3.3.0 \
        safetensors==0.7.0 && \
    pip install --no-cache-dir --no-deps rerun-sdk==0.26.2 lerobot==0.4.3

# LeRobot ACT instantiates a torchvision ResNet-18 visual backbone before it
# loads local checkpoint weights. Cache the ImageNet weights at build time so
# inference does not require network access.
RUN python -c "from torchvision import models; models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)"

COPY server.py policy.py ./

# Expected layout (must be provided at build time):
#   /app/policies/classifier/garment_classifier.pt
#   /app/policies/specialists/{top_long,top_short,pant_long,pant_short}/pretrained_model/
#   /app/policies/unified/pretrained_model/
#   /app/policies/datasets/{top_long,top_short,pant_long,pant_short}_merged/meta/
#   /app/policies/datasets/four_types_merged/meta/
COPY policies /app/policies

ENV POLICIES_ROOT=/app/policies

EXPOSE 8080
CMD ["python", "policy.py"]
