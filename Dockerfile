# circuit-forge: spec YAML -> synthesized, ngspice-verified circuit + HTML report.
#
# Build:
#   docker build -t cforge .
#
# Run (mount your specs and an output directory):
#   docker run --rm -v "$PWD/examples:/work/examples:ro" -v "$PWD/out:/work/out" \
#     cforge design examples/lp_1k.yaml -o out --no-mc
#
# Environment check:
#   docker run --rm cforge check-env

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# ngspice is a hard requirement: every numeric verdict comes from it.
# ca-certificates is needed for pip; libgomp for scipy/numpy on slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ngspice \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ngspice -v | head -n 5

WORKDIR /work

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY scripts ./scripts
COPY patterns ./patterns
COPY models ./models
COPY templates ./templates
COPY examples ./examples

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -e . \
    && python scripts/check_env.py

# Default command shows help. Override with `design`, `list-patterns`, etc.
ENTRYPOINT ["cforge"]
CMD ["--help"]
