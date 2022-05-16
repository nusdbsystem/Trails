# Envs

```bash
unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/project/TRAILS/internal/ml/
export PYTHONPATH=$PYTHONPATH:/project/TRAILS/internal/ml/model_slicing
echo $PYTHONPATH
```

# Save data 

4 datasets are used here.

```
adult  bank  cvd  frappe
```

Save the statistics

```bash
# save the data cardinalities, run in docker

# frappe
python3 ./internal/ml/model_slicing/save_satistics.py --dataset frappe --data_dir /hdd1/sams/data/ --nfeat 5500 --nfield 10 --max_filter_col 10 --train_dir ./

# adult
python3 ./internal/ml/model_slicing/save_satistics.py --dataset adult --data_dir /hdd1/sams/data/ --nfeat 140 --nfield 13 --max_filter_col 13 --train_dir ./

# cvd
python3 ./internal/ml/model_slicing/save_satistics.py --dataset cvd --data_dir /hdd1/sams/data/ --nfeat 110 --nfield 11 --max_filter_col 11 --train_dir ./

# bank
python3 ./internal/ml/model_slicing/save_satistics.py --dataset bank --data_dir /hdd1/sams/data/ --nfeat 80 --nfield 16 --max_filter_col 16 --train_dir ./
```

# Run docker

```bash
# in server
ssh panda17

# goes to /home/xingnaili/firmest_docker/TRAILS
git submodule update --recursive --remote

# run container
docker run -d --name trails \
  --network="host" \
  -v $(pwd)/TRAILS:/project/TRAILS \
  -v /hdd1/xingnaili/exp_data/:/project/exp_data \
  -v /hdd1/sams/tensor_log/:/project/tensor_log \
  -v /hdd1/sams/data/:/project/data_all \
  trails
    
# Enter the docker container.
docker exec -it trails bash 
```

# Run in database

Config the database runtime

```sql
cargo pgrx run
```

Load data into RDBMS

```bash

psql -h localhost -p 28814 -U postgres 
\l
\c pg_extension
\dt
\d frappe_train


# frappe
bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/data_all/frappe frappe
# frappe, only feature ids
bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db_int.sh /project/data_all/frappe frappe


# adult
bash ./internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/data_all/adult adult
# adult, only feature ids
bash ./internal/ml/model_selection/scripts/database/load_data_to_db_int.sh /project/data_all/adult adult
# check type is correct or not. 
SELECT column_name, data_type, column_default, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'adult_int_train';


# cvd 
bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/data_all/cvd cvd
# cvd, only feature ids
bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db_int.sh /project/data_all/cvd cvd


# bank
bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/data_all/bank bank
# bank, only feature ids
bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db_int.sh /project/data_all/bank bank
```

Verify data is in the DB

```sql
# check table status
\dt
\d frappe_train
SELECT * FROM frappe_train LIMIT 10;
```

Config

```sql
# after run the pgrx, then edie the sql
# generate schema
cargo pgrx schema >> /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql


-- src/lib.rs:266
-- pg_extension::sams_model_init
CREATE  FUNCTION "sams_model_init"(
	"condition" TEXT, /* alloc::string::String */
	"config_file" TEXT, /* alloc::string::String */
	"col_cardinalities_file" TEXT, /* alloc::string::String */
	"model_path" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE 
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'sams_model_init_wrapper';

-- src/lib.rs:242
-- pg_extension::sams_inference_shared_write_once_int
CREATE  FUNCTION "sams_inference_shared_write_once_int"(
	"dataset" TEXT, /* alloc::string::String */
	"condition" TEXT, /* alloc::string::String */
	"config_file" TEXT, /* alloc::string::String */
	"col_cardinalities_file" TEXT, /* alloc::string::String */
	"model_path" TEXT, /* alloc::string::String */
	"sql" TEXT, /* alloc::string::String */
	"batch_size" INT /* i32 */
) RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE 
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'sams_inference_shared_write_once_int_wrapper';

-- src/lib.rs:219
-- pg_extension::sams_inference_shared_write_once
CREATE  FUNCTION "sams_inference_shared_write_once"(
	"dataset" TEXT, /* alloc::string::String */
	"condition" TEXT, /* alloc::string::String */
	"config_file" TEXT, /* alloc::string::String */
	"col_cardinalities_file" TEXT, /* alloc::string::String */
	"model_path" TEXT, /* alloc::string::String */
	"sql" TEXT, /* alloc::string::String */
	"batch_size" INT /* i32 */
) RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE 
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'sams_inference_shared_write_once_wrapper';

-- src/lib.rs:196
-- pg_extension::sams_inference_shared
CREATE  FUNCTION "sams_inference_shared"(
	"dataset" TEXT, /* alloc::string::String */
	"condition" TEXT, /* alloc::string::String */
	"config_file" TEXT, /* alloc::string::String */
	"col_cardinalities_file" TEXT, /* alloc::string::String */
	"model_path" TEXT, /* alloc::string::String */
	"sql" TEXT, /* alloc::string::String */
	"batch_size" INT /* i32 */
) RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE 
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'run_sams_inference_shared_wrapper';

-- src/lib.rs:173
-- pg_extension::sams_inference
CREATE  FUNCTION "sams_inference"(
	"dataset" TEXT, /* alloc::string::String */
	"condition" TEXT, /* alloc::string::String */
	"config_file" TEXT, /* alloc::string::String */
	"col_cardinalities_file" TEXT, /* alloc::string::String */
	"model_path" TEXT, /* alloc::string::String */
	"sql" TEXT, /* alloc::string::String */
	"batch_size" INT /* i32 */
) RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE 
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'run_sams_inference_wrapper';


# record the necessary func above and then copy it to following
rm /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql
vi /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql

# then drop/create extension
DROP EXTENSION IF EXISTS pg_extension;
CREATE EXTENSION pg_extension;

pip install einops
```

Examples

```sql

# this is database name, columns used, time budget, batch size, and config file
SELECT count(*) FROM frappe_train WHERE col2='973:1' LIMIT 1000;
SELECT col2, count(*) FROM frappe_train group by col2 order by count(*) desc;

# query with two conditions
SELECT sams_inference(
    'frappe', 
    '{"1":266, "2":1244}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    'WHERE col1=''266:1'' and col2=''1244:1''', 
    32
);

# query with 1 conditions
SELECT sams_inference(
    'frappe', 
    '{"2":977}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    'WHERE col2=''977:1''', 
    10000
); 

# query with no conditions
SELECT sams_inference(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    8000
); 

# explaination
EXPLAIN (ANALYZE, BUFFERS) SELECT sams_inference(
    'frappe', 
    '{"2":977}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    'WHERE col2=''977:1''', 
    8000
); 


```

# Clear cache

```sql
DISCARD ALL;
```

# Benchmark Latency over all datasets

## Adult

```sql
SELECT sams_inference(
    'adult', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5', 
    '', 
    10000
); 


# exps
SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5', 
); 
SELECT sams_inference(
    'adult', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5', 
    '', 
    10000
); 


SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5'
); 
SELECT sams_inference_shared_write_once(
    'adult', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5', 
    '', 
    100000
); 

# replicate data 
INSERT INTO adult_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13
FROM adult_train;

INSERT INTO adult_int_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13
FROM adult_int_train;
```

## Frappe

```sql
SELECT sams_inference(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    10000
); 

SELECT sams_inference(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    20000
); 


SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4'
); 
SELECT sams_inference(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    10000
); 



SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4'
); 
SELECT sams_inference_shared_write_once(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 


SELECT sams_inference_shared(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    40000
); 



SELECT sams_inference(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    80000
); 


SELECT sams_inference(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    160000
); 

# replicate data 
INSERT INTO frappe_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10
FROM frappe_train;


INSERT INTO frappe_int_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10
FROM frappe_int_train;
```

## CVD

```sql
SELECT sams_inference(
    'cvd', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5', 
    '', 
    10000
); 

# exps
SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5', 
); 
SELECT sams_inference(
    'cvd', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5', 
    '', 
    10000
); 


SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5'
); 
SELECT sams_inference_shared_write_once(
    'cvd', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5', 
    '', 
    100000
); 


# replicate data 
INSERT INTO cvd_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 
FROM cvd_train;

INSERT INTO cvd_int_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 
FROM cvd_int_train;

```

## Bank

```sql
SELECT sams_inference(
    'bank', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3', 
    '', 
    10000
); 


# exps
SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3', 
); 
SELECT sams_inference(
    'bank', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3', 
    '', 
    10000
); 


SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3'
); 
SELECT sams_inference_shared_write_once(
    'bank', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3', 
    '', 
    100000
); 


# replicate data 
INSERT INTO bank_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16 
FROM bank_train;


INSERT INTO bank_int_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16 
FROM bank_int_train;

```

# Baseline & SAMS

## Frappe

```bash
# frappe
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline.py /hdd1/sams/tensor_log/frappe/dnn_K16_alpha4 --device cpu --dataset frappe --batch_size 10 --col_cardinalities_file frappe_col_cardinalities --target_batch 10


CUDA_VISIBLE_DEVICES="0" python ./internal/ml/model_slicing/baseline.py /hdd1/sams/tensor_log/frappe/dnn_K16_alpha4 --device cuda:0 --dataset frappe --batch_size 100000 --col_cardinalities_file frappe_col_cardinalities --target_batch 100000

CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline_int.py /hdd1/sams/tensor_log/frappe/dnn_K16_alpha4 --device cpu --dataset frappe --batch_size 100000 --col_cardinalities_file frappe_col_cardinalities --target_batch 100000


SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4'
); 
SELECT sams_inference_shared_write_once(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 

# read int data
SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4'
); 
SELECT sams_inference_shared_write_once_int(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 
```

## Adult

```bash

# adult
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline.py /hdd1/sams/tensor_log/adult/Ednn_K16_alpha2-5 --device cpu --dataset adult --batch_size 100000 --col_cardinalities_file adult_col_cardinalities  --target_batch 100000

CUDA_VISIBLE_DEVICES="0" python ./internal/ml/model_slicing/baseline.py /hdd1/sams/tensor_log/adult/Ednn_K16_alpha2-5 --device cuda:0 --dataset adult --batch_size 100000 --col_cardinalities_file adult_col_cardinalities  --target_batch 100000

CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline_int.py /hdd1/sams/tensor_log/adult/Ednn_K16_alpha2-5 --device cpu --dataset adult --batch_size 100000 --col_cardinalities_file adult_col_cardinalities  --target_batch 100000

SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5'
); 
SELECT sams_inference_shared_write_once(
    'adult', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5', 
    '', 
    100000
); 

SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5'
); 
SELECT sams_inference_shared_write_once_int(
    'adult', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5', 
    '', 
    640000
); 
```

## CVD
```bash
# CVD
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline.py /hdd1/sams/tensor_log/cvd/dnn_K16_alpha2-5 --device cpu --dataset cvd --batch_size 100000 --col_cardinalities_file cvd_col_cardinalities  --target_batch 100000

CUDA_VISIBLE_DEVICES="0" python ./internal/ml/model_slicing/baseline.py /hdd1/sams/tensor_log/cvd/dnn_K16_alpha2-5 --device cuda:0 --dataset cvd --batch_size 100000 --col_cardinalities_file cvd_col_cardinalities  --target_batch 100000

CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline_int.py /hdd1/sams/tensor_log/cvd/dnn_K16_alpha2-5 --device cpu --dataset cvd --batch_size 100000 --col_cardinalities_file cvd_col_cardinalities  --target_batch 100000


SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5'
); 
SELECT sams_inference_shared_write_once(
    'cvd', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5', 
    '', 
    100000
); 

SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5'
); 
SELECT sams_inference_shared_write_once_int(
    'cvd', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5', 
    '', 
    100000
); 
```

## Bank

```bash
# Bank
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline.py /hdd1/sams/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3 --device cpu --dataset bank --batch_size 100000 --col_cardinalities_file bank_col_cardinalities  --target_batch 100000

CUDA_VISIBLE_DEVICES="0" python ./internal/ml/model_slicing/baseline.py /hdd1/sams/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3 --device cuda:0 --dataset bank --batch_size 100000 --col_cardinalities_file bank_col_cardinalities  --target_batch 100000


CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline_int.py /hdd1/sams/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3 --device cpu --dataset bank --batch_size 100000 --col_cardinalities_file bank_col_cardinalities  --target_batch 100000

SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3'
); 
SELECT sams_inference_shared_write_once(
    'bank', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3', 
    '', 
    100000
); 


SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3'
); 
SELECT sams_inference_shared_write_once_int(
    'bank', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3', 
    '', 
    100000
); 


```

# Micro

## Profiling

```bash
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline.py /hdd1/sams/tensor_log/frappe/dnn_K16_alpha4 --device cpu --dataset frappe --batch_size 20000 --col_cardinalities_file frappe_col_cardinalities --target_batch 20000`
```

## Optimizations

```bash

# 1. with all opt
SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4'
); 
SELECT sams_inference_shared_write_once(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 

# 2. w/o model cache
SELECT sams_inference_shared_write_once(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 

# 3. w/o shared memory
SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4'
); 
SELECT sams_inference(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 

# w/o SPI this can measure the time usage for not using spi
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline.py /hdd1/sams/tensor_log/frappe/dnn_K16_alpha4 --device cpu --dataset frappe --batch_size 100000 --col_cardinalities_file frappe_col_cardinalities --target_batch 100000
```

Int dataset

```bash

# 1. with all opt
SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4'
); 
SELECT sams_inference_shared_write_once_int(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 

# 2. w/o model cache
SELECT sams_inference_shared_write_once_int(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 

# 3. w/o shared memory
SELECT sams_model_init(
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4'
); 
SELECT sams_inference(
    'frappe', 
    '{}', 
    '/project/TRAILS/internal/ml/model_selection/config.ini', 
    '/project/TRAILS/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 

# w/o SPI this can measure the time usage for not using spi
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline.py /hdd1/sams/tensor_log/frappe/dnn_K16_alpha4 --device cpu --dataset frappe --batch_size 100000 --col_cardinalities_file frappe_col_cardinalities --target_batch 100000
```













