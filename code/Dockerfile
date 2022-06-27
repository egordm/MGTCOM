FROM mambaorg/micromamba:0.24.0

ARG ENV_FILE=environment.yml
COPY --chown=$MAMBA_USER:$MAMBA_USER $ENV_FILE /tmp/env.yml
COPY datasets/requirements.txt datasets/requirements.txt
COPY ml/requirements.txt ml/requirements.txt


#RUN --mount=type=cache,target=/opt/conda/pkgs/cache micromamba install -y -n base -f /tmp/env.yml && \
#    micromamba clean --all --yes \
RUN micromamba install -y -n base -f /tmp/env.yml && micromamba clean --all --yes

RUN mkdir /tmp/numba_cache && chmod 777 /tmp/numba_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app
ENV PYTHONPATH=/app/shared:/app/datasets:/app/ml:$PYTHONPATH
COPY . .
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python"]
