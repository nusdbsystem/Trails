FROM ubuntu:20.04

# Install Python, Vim, and necessary libraries
RUN apt-get update && \
    apt-get install -y software-properties-common wget gnupg2 lsb-release git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.6 python3-pip vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install necessary dependencies for PostgreSQL and Rust
RUN apt-get update && \
    apt-get install -y pkg-config libssl-dev libpq-dev libclang-dev curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PostgreSQL
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - && \
    echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list && \
    apt-get update && \
    apt-get install -y  \
    postgresql-14 \
    postgresql-server-dev-14 \
    postgresql-plpython3-14 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Rust and init the cargo
USER postgres
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    echo 'source $HOME/.cargo/env' >> $HOME/.bashrc && \
    /bin/bash -c "source $HOME/.cargo/env && cargo install cargo-pgrx --version '0.9.7' --locked" && \
    /bin/bash -c "source $HOME/.cargo/env && cargo pgrx init --pg14 /usr/bin/pg_config"

# Run as root
USER root
RUN mkdir /project && \
    adduser postgres sudo && \
    chown -R postgres:postgres /project && \
    chown -R postgres:postgres /usr/share/postgresql/ && \
    chown -R postgres:postgres /usr/lib/postgresql/ && \
    chown -R postgres:postgres /var/lib/postgresql/

# Switch to the postgres user and run rdbms
USER postgres

# Set environment variables for Rust and Python
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PYTHONPATH="${PYTHONPATH}:/project/TRAILS/internal/ml/model_selection"

# Set environment variables for PostgreSQL
ENV PGDATA /var/lib/postgresql/data

WORKDIR /project
COPY ./internal/ml/model_selection/requirement.txt ./requirement.txt
RUN pip install -r requirement.txt

# Initialize PostgreSQL data directory
RUN mkdir -p ${PGDATA} && chown -R postgres:postgres ${PGDATA} && \
    service postgresql start && \
    /usr/lib/postgresql/14/bin/initdb -D ${PGDATA}

# CMD statement to start PostgreSQL when the container starts
CMD service postgresql start && tail -F /var/log/postgresql/postgresql-14-main.log
