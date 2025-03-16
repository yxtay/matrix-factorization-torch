##
# base
##
FROM python:3.13-slim@sha256:f3614d98f38b0525d670f287b0474385952e28eb43016655dd003d0e28cf8652 AS base
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
        curl=7.88.1-10+deb12u8 \
    && rm -rf /var/lib/apt/lists/*

ARG PYTHONDONTWRITEBYTECODE=1
ARG UV_NO_CACHE=1

# set up python
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:031ddbc79275e351a43cbb66f64d8cd314cc78c3878898f4ab4f147b092e8e2d /uv /uvx /bin/
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
