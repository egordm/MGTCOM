FROM gcc:11.1-bullseye AS build

WORKDIR /build

COPY louvain-generic ./louvain-generic
RUN cd louvain-generic && make

COPY . .
RUN make all


FROM debian:bullseye-slim

WORKDIR /app

COPY --from=build \
    /build/louvain-generic/convert \
    /build/louvain-generic/hierarchy \
    /build/louvain-generic/louvain \
    /build/louvain-generic/matrix \
    /app/louvain-generic/

COPY --from=build \
    /build/MOSES/moses-binary-linux-32bit \
    /build/MOSES/moses-binary-linux-x86-64 \
    /build/MOSES/moses-binary-Mach-O-i386 \
    /app/MOSES/

COPY --from=build \
    /build/aggregator \
    /build/node_stats \
    /build/step_stats \
    /build/timeline_stats \
    /build/tracker \
    /app/

RUN ln -s /app/louvain-generic/louvain ./louvain \
    && ln -s /app/MOSES/moses-binary-linux-x86-64 ./moses

ENTRYPOINT ["/app/moses"]