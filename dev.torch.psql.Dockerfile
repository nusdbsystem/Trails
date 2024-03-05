FROM ubuntu:20.04

#LABEL maintainer="Naili Xing <xingnaili14@gmai.com>"

ENV DEBIAN_FRONTEND=noninteractive

# Install Python, Vim, and necessary libraries
RUN apt-get update && \
    apt-get install -y software-properties-common wget gnupg2 lsb-release git sudo && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.6 python3-pip vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install necessary dependencies for PostgreSQL and Rust
RUN apt-get update && \
    apt-get install -y pkg-config libssl-dev libpq-dev libclang-dev curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install necessary dependencies for pgrx
RUN apt-get update && \
    apt-get install -y bison flex libreadline-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create the postgres user
USER root
RUN adduser --disabled-password --gecos "" postgres && \
    mkdir /project && \
    adduser postgres sudo && \
    chown -R postgres:postgres /project

# Add PostgreSQL's repository
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - \
    && sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(. /etc/os-release; echo $VERSION_CODENAME)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

# Install postgresql client
RUN apt-get update && apt-get install -y \
    postgresql-client-14 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch to the postgres user and Install Rust and init the cargo
USER postgres
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    echo 'source $HOME/.cargo/env' >> $HOME/.bashrc && \
    /bin/bash -c "source $HOME/.cargo/env && cargo install cargo-pgrx --version '0.9.7' --locked" && \
    /bin/bash -c "source $HOME/.cargo/env && cargo pgrx init"

# Set environment variables for Rust and Python
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PYTHONPATH="${PYTHONPATH}:/project/Trails/internal/ml/model_selection"
ENV PYTHONPATH="${PYTHONPATH}:/project/Trails/internal/ml/model_slicing"
ENV PYTHONPATH="${PYTHONPATH}:/project/Trails/internal/ml/model_slicing/algorithm"

# Clone code to there, install dependences,
CMD ["tail", "-f", "/dev/null"]
