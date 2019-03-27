ARG BASE_IMAGE

FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY ./app.pex /app/bin/app.pex

CMD ["/app/bin/app.pex"]