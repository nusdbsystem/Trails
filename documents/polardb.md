

# Env

panda 16

# PolarDB Start

```bash
# pull polarDB image
docker pull polardb/polardb_pg_local_instance


# run contrainer
docker run -d --rm --name polardb_instance \
  --network="host" \
  -v /hdd1/sams/:/home/postgres/dataset/ polardb/polardb_pg_local_instance
  
docker run -d --rm --name polardb_instance polardb/polardb_pg_local_instance
```

# Load dataset

```bash
# enter posgresql
docker exec -it polardb_instance bash

# enter primary node, where the "transaction_read_only" == "off"
# or verify by executing SHOW transaction_read_only;
psql -h localhost -p 5432 -U postgres 
```

then load dataset

```sql
CREATE DATABASE model_slicing;

CREATE TABLE iris (
    id SERIAL PRIMARY KEY,
    sepal_length VARCHAR,
    sepal_width VARCHAR,
    petal_length VARCHAR,
    petal_width VARCHAR,
    species VARCHAR
);

# load data into it
\COPY iris(sepal_length, sepal_width, petal_length, petal_width, species) FROM '/home/postgres/dataset/data/iris/iris.csv' DELIMITER ',' CSV HEADER NULL AS '';
```

# Usage

```sql
select * from iris limit 10; 

CREATE OR REPLACE FUNCTION get_iris_data()
RETURNS SETOF iris AS $$
BEGIN
    RETURN QUERY SELECT * FROM iris;
END;
$$ LANGUAGE plpgsql;


SELECT * FROM get_iris_data();

SELECT * FROM get_iris_data() limit 10;


psql -h localhost -p 5432 -U postgres 

DROP EXTENSION IF EXISTS pg_extension;
CREATE EXTENSION pg_extension;

# Test coordinator
SELECT coordinator('0.08244', '168.830156', '800', false, '/home/postgres/Trails/internal/ml/model_selection/config.ini');

/home/postgres/Trails

export PYTHONPATH=$PYTHONPATH:/home/postgres/Trails/internal/ml/model_selection/
export PATH=$PATH:/home/postgres/Trails/internal/ml/model_selection/
echo $PYTHONPATH
echo $PATH
echo $PYTHONHOME


export HOME=/home/project
```

# Config the python

```bash
sudo apt-get install pip
sudo apt-get install bzip2 libbz2-dev
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.8
pyenv virtualenv pyo3
```

# Config on the pgrx

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
source $HOME/.cargo/env && cargo install cargo-pgrx --version '0.9.7' --locked

https://github.com/NLGithubWP/Trails.git

cargo pgrx init --pg11 /home/postgres/tmp_basedir_polardb_pg_1100_bld/bin/pg_config
# Creating PGRX_HOME at `/home/postgres/.pgrx`
# Validating /home/postgres/tmp_basedir_polardb_pg_1100_bld/bin/pg_config
# Initializing data directory at /home/postgres/.pgrx/data-11
```

```bash
change the default version in Cargo.toml as pg11
cargo clean
cargo pgrx install --pg-config /home/postgres/tmp_basedir_polardb_pg_1100_bld/bin/pg_config


Using PgConfig("pg11") and `pg_config` from /home/postgres/tmp_basedir_polardb_pg_1100_bld/bin/pg_config
    Building extension with features python pg11
     Running command "/home/postgres/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo" "build" "--features" "python pg11" "--no-default-features" "--message-format=json-render-diagnostics"
   Compiling pg_extension v0.1.0 (/home/postgres/Trails/internal/pg_extension)
    Finished dev [unoptimized + debuginfo] target(s) in 7.65s
  Installing extension
     Copying control file to /home/postgres/tmp_basedir_polardb_pg_1100_bld/share/extension/pg_extension.control
     Copying shared library to /home/postgres/tmp_basedir_polardb_pg_1100_bld/lib/pg_extension.so
 Discovering SQL entities
```

