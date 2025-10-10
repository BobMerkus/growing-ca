FROM docker.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

ENV UV_LINK_MODE=copy
ENV UV_SYSTEM_PYTHON=1

RUN --mount=type=cache,target=/var/cache/apt \
    set -eu \
    && apt-get update \
    && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential

RUN --mount=type=cache,target=/var/cache/apt \
    set -eu \
    && apt-get install -y \
    net-tools inetutils-ping curl


# Create user first
RUN printf 'CREATE_MAIL_SPOOL=no' >> /etc/default/useradd \
    && mkdir -p /home/runner \
    && groupadd runner \
    && useradd runner -g runner -d /home/runner \
    && chown runner:runner /home/runner

# Install jupyter
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install jupyter matplotlib --system

# Copy the project into the image
ADD . /home/runner/app/
WORKDIR /home/runner/app/

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install . --system

WORKDIR /home/runner
USER runner:runner
VOLUME /home/runner
EXPOSE 8888
ENV CLI_ARGS=""
CMD ["bash","/home/runner/entrypoint.sh"]