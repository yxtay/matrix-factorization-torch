##
# base
##
FROM python:3.12-slim@sha256:c4c4ded064fdc28eab54fe119a0778c8a7915fc5b713497ff4cc5fdcab0dacaa AS base
LABEL maintainer="wyextay@gmail.com"

# set up user
ARG USER=user
ARG UID=1000
RUN useradd --create-home --shell /bin/false --uid ${UID} ${USER}

# set up environment
ARG APP_HOME=/work/app
ARG VIRTUAL_ENV=${APP_HOME}/.venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=${VIRTUAL_ENV}

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
    && rm -rf /var/lib/apt/lists/*

ARG PYTHONDONTWRITEBYTECODE=1
ARG UV_NO_CACHE=1

# set up python
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:4a6c9444b126bd325fba904bff796bf91fb777bf6148d60109c4cb1de2ffc497 /uv /uvx /bin/
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
HEALTHCHECK CMD [ uv, run, --no-sync, lightning, fit, --print_config ]
