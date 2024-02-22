

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
```

