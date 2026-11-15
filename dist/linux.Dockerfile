FROM rust:1.96.0-slim AS rust
FROM ubuntu:22.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates pkg-config clang libclang-dev python3 && rm -rf /var/lib/apt/lists/*
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
ENV PATH=/usr/local/cargo/bin:$PATH CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup CARGO_TARGET_DIR=/target
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY vendor ./vendor
COPY dist ./dist
COPY scripts/install-slang ./scripts/install-slang
RUN python3 scripts/install-slang /opt/slang
ENV VORK_SLANGC=/opt/slang/bin/slangc
COPY docs ./docs
COPY examples ./examples
COPY libraries ./libraries
COPY skills ./skills
ARG VORK_BUILD_COMMIT
RUN test -n "$VORK_BUILD_COMMIT" && VORK_BUILD_COMMIT="$VORK_BUILD_COMMIT" cargo build --locked --release -p vork-cli
RUN objcopy --only-keep-debug /target/release/vork /target/release/vork.debug && strip --strip-unneeded /target/release/vork
RUN /target/release/vork --version && objdump -T /target/release/vork | sed -n 's/.*(GLIBC_\([0-9.]*\)).*/\1/p' | sort -Vu | tail -1
FROM ubuntu:22.04 AS smoke
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libgcc-s1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /target/release/vork /usr/local/bin/vork
RUN vork --version
FROM scratch AS artifact
COPY --from=smoke /usr/local/bin/vork /vork
COPY --from=build /target/release/vork-slang-filesystem /vork-slang-filesystem
COPY --from=build /src/crates/vork-render/src/ltc/LICENSE.ltc /licenses/LICENSE.ltc
COPY --from=build /src/crates/vork-render/src/ltc/LICENSE.three /licenses/LICENSE.three
COPY --from=build /src/crates/vork-render/src/ltc/README.md /licenses/LTC.md
