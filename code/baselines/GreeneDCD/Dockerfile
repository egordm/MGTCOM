FROM gcc:11.1-bullseye AS build

WORKDIR /build

COPY louvain-generic ./louvain-generic
RUN cd louvain-generic && make

COPY common ./common
COPY *.cpp *.h *.c Makefile ./
RUN make all


FROM python:3.9-slim

WORKDIR /app

COPY --from=build \
    /build/louvain-generic/convert \
    /build/louvain-generic/hierarchy \
    /build/louvain-generic/louvain \
    /build/louvain-generic/matrix \
    /app/louvain-generic/

COPY \
    ./MOSES/moses-binary-linux-32bit \
    ./MOSES/moses-binary-linux-x86-64 \
    ./MOSES/moses-binary-Mach-O-i386 \
    /app/MOSES/

COPY --from=build \
    /build/aggregator \
    /build/node_stats \
    /build/step_stats \
    /build/timeline_stats \
    /build/tracker \
    /app/

RUN ln -s /app/MOSES/moses-binary-linux-x86-64 /app/MOSES/moses

COPY run.py .

ENTRYPOINT ["python", "run.py"]