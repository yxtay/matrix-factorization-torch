##
# base
##
FROM python:3.12-slim@sha256:aaa3f8cb64dd64e5f8cb6e58346bdcfa410a108324b0f28f1a7cc5964355b211 AS base
LABEL maintainer="wyextay@gmail.com"

# set up user
ARG USER=user
ARG UID=1000
RUN useradd --create-home --shell /bin/false --uid ${UID} ${USER}

# set up environment
ARG APP_HOME=/work/app
ARG VIRTUAL_ENV=${APP_HOME}/.venv
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=${VIRTUAL_ENV} \
    PATH=${VIRTUAL_ENV}/bin:${PATH}

WORKDIR ${APP_HOME}

##
# dev
##
FROM base AS dev

ARG DEBIAN_FRONTEND=noninteractive
COPY <<-EOF /etc/apt/apt.conf.d/99-disable-recommends
APT::Install-Recommends "false";
APT::Install-Suggests "false";
APT::AutoRemove::RecommendsImportant "false";
APT::AutoRemove::SuggestsImportant "false";
EOF

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        build-essential=12.9 \
        curl=7.88.1-10+deb12u8 \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_NO_COMPILE=0 \
    PIP_NO_INPUT=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1

# set up python
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:562193a4a9d398f8aedddcb223e583da394ee735de36b5815f8f1d22cb49be15 /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --seed "${VIRTUAL_ENV}" && \
    uv sync --frozen --no-default-groups --no-install-project && \
    chown -R "${USER}:${USER}" "${VIRTUAL_ENV}" && \
    chown -R "${USER}:${USER}" "${APP_HOME}" && \
    uv pip list

# set up project
COPY mf_torch mf_torch
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups

USER ${USER}
HEALTHCHECK CMD [ uv, run, lightning, fit, --print_config ]
