FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.txt .
RUN grep -v -E "^torch" requirements.txt > /tmp/requirements_clean.txt && \
    pip install --no-cache-dir -r /tmp/requirements_clean.txt && \
    pip install --no-cache-dir google-cloud-storage google-cloud-batch tensorboard && \
    pip install --no-cache-dir flash-attn

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgmp10 libgmpxx4ldbl libmpfr6 \
    && rm -rf /var/lib/apt/lists/*

COPY cgalpy-1.0.4-cp312-cp312-linux_x86_64.whl .
RUN pip3 install --no-cache-dir cgalpy-1.0.4-cp312-cp312-linux_x86_64.whl

COPY . .
RUN pip install -e .

RUN chmod +x run/cloud/entrypoint.sh
ENTRYPOINT ["/bin/bash", "run/cloud/entrypoint.sh"]
