# How to Reproduce the results

# Config Environments

```bash
# Create a virtual env
conda config --set ssl_verify false
conda create -n "trails" python=3.8.10
conda activate trails
pip install -r requirement.txt

cd TRAILS
export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
# make a dir to store all results.
mkdir ../exp_data
git submodule update --init
```

# Reproduce the results

## NAS-Bench-Tabular

 NAS-Bench-Tabular can be either **downloaded** or built from scratch.

### Download NAS-Bench-Tabular

1. **Download** the dataset using the following link, and extract them to `exp_data`

```bash
https://drive.google.com/file/d/1TGii9ymbmX81c9-GKWXbe_4Z64R8Btz1/view?usp=sharing
```

### Build NAS-Bench-Tabular

2. Build the **NAS-Bench-Tabular** from scratch

```python
# Construct NAS-Bench-Tabular:
## 1. Training all models.
bash internal/ml/model_selection/scripts/nas-bench-tabular/train_all_models_frappe.sh
bash internal/ml/model_selection/scripts/nas-bench-tabular/train_all_models_diabetes.sh
bash internal/ml/model_selection/scripts/nas-bench-tabular/train_all_models_criteo.sh

## 2. Scoring all models using all TFMEMs.
bash internal/ml/model_selection/scripts/nas-bench-tabular/score_all_modesl_frappe.sh
bash internal/ml/model_selection/scripts/nas-bench-tabular/score_all_modesl_uci.sh
bash internal/ml/model_selection/scripts/nas-bench-tabular/score_all_modesl_criteo.sh
```

## Build the **NAS-Bench-Img** from scratch

To facilitate the experiments and query speed (NASBENCH API is slow)

1. We retrieve all results from NASBENCH API and store them as a JSON file.
2. We score all models in NB201 and 28K models in NB101.
3. We search with  EA + Score and record the search process in terms of
    `run_id,  current_explored_model, top_400 highest scored model, time_usage`
     to SQLite.

```python
# 1. Record NASBENCH API data into a json file
## This requires to install nats_bench: pip install nats_bench
bash ./internal/ml/model_selection/scripts/nas-bench-img/convert_api_2_json.sh

# 2. Scoring all models using all TFMEMs.
nohup bash ./internal/ml/model_selection/scripts/nas-bench-img/score_all_models.sh &

# 3. Explore with EA and score results and store exploring process into SQLLite
bash ./internal/ml/model_selection/scripts/nas-bench-img/explore_all_models.sh

# 4. Generate the baseline.
bash ./internal/ml/model_selection/scripts/baseline_system_img.sh
```

The following experiment could then query filtering phase results based on `run_id`.

## **Design 2Phase Alg**

1. Search Spaces

   Global value, Mediam value,

   ECDF + Parameter/AUC + Benchmark Search Strategy.

   Decide which Epoch's value to compare.

   ```bash
   Frappe: 14, UCI_Diabalte: 1, Crito: 20.
   ```

2. Ablation study of correlation

   Each dataset, Parameter Posivity -> Parameter initialization -> Batch size -> Batch data -> Depth&Width

   positive -> He -> 4 -> 1-> ?

   Present all correlations.

3. With scoring methods, study the second phase, and choose a few algorithms.

   SH, SJ, UA -> Sh is best

4. Coordinator

   K vs U -> U = 1 is better, and K is large is better.

   K vs N -> N = 13K is OK.

5. Two-phase end-end

   Compare with TabNAS, EA-NAS.

## Macro 1: Effective Combinations

```bash
# Generate result
python ./internal/ml/model_selection/exps/micro/benchmark_train_free_train_based_combines_phase_query.py
# Copy the last sentence (result), then draw graph
python ./internal/ml/model_selection/exps/micro/draw_train_free_train_based_combines.py
```

## Macro 0: Effectiveness (SLO-Aware)

With the above **NAS-Bench-Tabular**, we could run various experiments.

```bash
# 1. Generate the results for drawing the figure for both image and tabular data
## tabular data: training-base-ms
bash internal/ml/model_selection/scripts/baseline_system_tab.sh
## tabular data: training-free-ms, 2phase-ms
nohup bash internal/ml/model_selection/scripts/anytime_tab.sh &
## image data: training-base-ms, training-free-ms, 2phase-ms
nohup bash internal/ml/model_selection/scripts/anytime_img_w_baseline.sh &

# 2. Add more baselines for tabular data, KNAS
export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
# run with this to get the knas resut. This is one exampel for Criteo
python3 ./internal/ml/model_selection/exps/baseline/knas.py --dataset criteo
# draw using the existing result.
python3 ./internal/ml/model_selection/exps/baseline/knas_simulate.py --dataset frappe
python3 ./internal/ml/model_selection/exps/baseline/knas_simulate.py --dataset criteo
python3 ./internal/ml/model_selection/exps/baseline/knas_simulate.py --dataset uci_diabetes

# 3. Draw figure
python internal/ml/model_selection/exps/macro/anytime_tab_draw_jacflow.py
python internal/ml/model_selection/exps/macro/anytime_tab_draw_expressflow.py
python internal/ml/model_selection/exps/macro/anytime_img_draw.py
```


## Macro 1: Efficiency (GPU Resource)

```bash
# Measure the hardware placement
python ./internal/ml/model_selection/exps/macro/cost_device_placement.py
# Measure the overall cost reduction
python ./internal/ml/model_selection/exps/macro/cost_high_end_time.py
```

## Macro 2: Efficiency (Query Latency)

```bash
# Measure the query latency of exploring various models
python ./internal/ml/model_selection/exps/macro/efficiency_workloads.py
# Measure the waiting time with/without cache service.
python ./internal/ml/model_selection/exps/macro/efficiency_wall_time.py
```

## Macro 3: Scalability

```bash
# CPU number
python ./internal/ml/model_selection/exps/macro/scale_filtering.py
# GPU number
python ./internal/ml/model_selection/exps/macro/scale_refinement.py
```

## Micro: Benchmark TFMEMs

```bash
export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
conda activate trails
python ./internal/ml/model_selection/exps/micro/benchmark_correlation.py
```

## Micro: Score and AUC relation

```bash
# get the score and AUC using our metrics on three datasets
python ./internal/ml/model_selection/exps/micro/benchmark_score_metrics.py
# draw the figure
python ./internal/ml/model_selection/exps/micro/draw_score_metric_relation.py
```

## Micro: Benchmark Budge-Aware Algorithm

We compare three budget-aware algs: `SH, SJ, and UA`. And examine how does they influence the efficieny and effectiveness.

```bash
bash internal/ml/model_selection/scripts/micro_budget_aware_alg.sh
```

## Micro: Benchmark N, K, U

By ranking the models by their TFMEM score in the filtering phase, we aim to determine

1. Is more models  (**K**) with each going through less training epoch (**U**) easier to find a good model? Or examine less models but each training more epochs?
2. How many models to explore (**N**) and how many to keep (**K**)?

```bash
bash internal/ml/model_selection/scripts/micro_nku_tradeoff.sh
```

This is the experimental result conducted at the UCI Diabetes datasets.
Clearly, exploring more models in the refinement phase (large **K** ) is more helpful in finding a better model.
Although increasing **U** can find a better model accurately, it runs more training epochs leading to higher training costs.

Then we fix **U=1** for cost efficiency and determine N/K for higher searching effectiveness.
Clearly, K/N reaching 100 yields better-scheduling results in both image and tabular datasets, thus, we set **N/K~100** in the coordinator.

## Micro: Device Placement & Embedding Cache

1. To measure the time usage for the filtering phase on various hardware, run the following

   ```bash
   # Without embedding cache at the filtering phase
   nohup bash internal/ml/model_selection/scripts/latency_phase1_cpu_gpu.sh &
   # With embedding cache at the filtering phase (faster)
   nohup bash internal/ml/model_selection/scripts/latency_embedding_cache.sh &
   # Draw graph
   python ./internal/ml/model_selection/exps/micro/draw_filtering_latency.py
   python ./internal/ml/model_selection/exps/micro/draw_filtering_memory_bar.py
   python ./internal/ml/model_selection/exps/micro/draw_filtering_memory_line.py
   python ./internal/ml/model_selection/exps/micro/draw_filtering_memory_cache_CPU.py
   ```

2. Further, we measure the end-to-end latency under two CPUs, GPU, and Hybrid.

   ```bash
   nohup bash internal/ml/model_selection/scripts/latency_phase1_cpu_gpu.sh &
   ```

## Micro: In-DB vs Out-DB filtering phase

Loda data into the database

```bash
bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/exp_data/data/structure_data/frappe frappe

bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/exp_data/data/structure_data/uci_diabetes uci_diabetes

bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/exp_data/data/structure_data/criteo_full criteo
```

Run SQL

```bash
# run out-of db, read data via psycopg2
export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
bash ./internal/ml/model_selection/scripts/latency_phase1_in_db.sh

# run in-db query, read data via SPI
select benchmark_filtering_latency_in_db(5000, 'frappe', 64,'/project/TRAILS/internal/ml/model_selection/config.ini');

select benchmark_filtering_latency_in_db(5000, 'uci_diabetes', 64,'/project/TRAILS/internal/ml/model_selection/config_uci.ini');

select benchmark_filtering_latency_in_db(10000, 'criteo', 64,'/project/TRAILS/internal/ml/model_selection/config_criteo.ini');

# draw the figure
python internal/ml/model_selection/exps/micro/draw_filtering_latency_cache_sql.py
```

Data Scalability

```sql
select benchmark_filtering_latency_in_db(500, 'criteo', 64,'/project/TRAILS/internal/ml/model_selection/config_criteo.ini');

select benchmark_filtering_latency_in_db(1000, 'criteo', 64,'/project/TRAILS/internal/ml/model_selection/config_criteo.ini');

select benchmark_filtering_latency_in_db(2000, 'criteo', 64,'/project/TRAILS/internal/ml/model_selection/config_criteo.ini');

select benchmark_filtering_latency_in_db(4000, 'criteo', 64,'/project/TRAILS/internal/ml/model_selection/config_criteo.ini');

# draw the figure
python internal/ml/model_selection/exps/micro/draw_filter_latency_scalability.py
```

## Micro: On-the-fly Data Transmission, Refinement

```bash
# start cache service
python ./internal/cache-service/cache_service.py
python ./internal/cache-service/trigger_cache_svc.py
```

## Micro: For large ImageNet

```bash
# Compute time score time
# for IN16
	python ./internal/ml/model_selection/exps/nas_bench_tabular/4.seq_score_online.py \
      --is_simulate False \
      --tfmem=jacflow \
      --models_explore=100 \
      --search_space=nasbench201 \
      --api_loc=NAS-Bench-201-v1_1-096897.pth \
      --base_dir=/hdd1/user_name/exp_data/ \
      --dataset=ImageNet16-120 \
      --batch_size=32 \
      --num_labels=120 \
      --device=cpu \
      --log_folder=log_score_all_img_imgnet \
      --result_dir=./
# for ImageNet1k
    python ./internal/ml/model_selection/exps/nas_bench_tabular/4.seq_score_online.py \
      --is_simulate False \
      --tfmem=jacflow \
      --models_explore=100 \
      --search_space=nasbench201 \
      --api_loc=NAS-Bench-201-v1_1-096897.pth \
      --base_dir=/hdd1/user_name/exp_data/ \
      --dataset=ImageNet1k \
      --batch_size=32 \
      --num_labels=1000 \
      --device=cpu \
      --log_folder=log_score_all_img_imgnet \
      --result_dir=./
# Above output has been recorded in score_imageNet_16x16 and score_imageNet_224x224
# run this to get the time staistics.
python ./internal/ml/model_selection/exps/micro/benchmark_time_large_imgnet.py

# Compute the train time epoch level, manually input batch size, 2, 4, 8, 16, 24.
# chagne the line 219 or 216 to change the device or image dataset
python ./internal/ml/model_selection/src/eva_engine/phase2/train_img_net.py


# SLO-aware on ImageNet
python ./internal/ml/model_selection/exps/macro/full_imgnet_slo.py
# draw the result with
python ./internal/ml/model_selection/exps/macro/full_imgnet_slo_draw.py
```

# Baselines

We compare with Training-Based MS, TabNAS, and training-free MS etc.

For image data, it already generated at the NAS-Bench-Img part, see above.

# Extra Exps

Here all experiments are on the Frappe dataset.

1. Sensitive Analysis

   ```bash
   # Impact of the parameter sign
   # change the code at evaluator.py, in mini_batch=new_model.generate_all_ones_embedding(32), here is 32 batch size.
   # then run those:
   bash internal/ml/model_selection/scripts/nas-bench-tabular/score_all_modesl_frappe.sh
   bash internal/ml/model_selection/scripts/nas-bench-tabular/score_all_modesl_uci.sh
   bash internal/ml/model_selection/scripts/nas-bench-tabular/score_all_modesl_criteo.sh
   ```

1. Computational Costs

   ```bash
   bash ./internal/ml/model_selection/exps/micro/resp/benchmark_cost.sh
   ```

2. Search Cost, multiple training-free or training-based combinations (warm-up / movel proposal)

   ```bash
   # get RL, RE, RS + training-based model evaluation
   bash ./internal/ml/model_selection/scripts/micro_search_strategy.sh
   # This will read previous file, and run warm-up/move proposal, and draw all together
   bash ./internal/ml/model_selection/exps/micro/resp/benchmark_search_cost.sh
   ```

3. How des the K influence the result?

   ```bash
   python ./internal/ml/model_selection/exps/micro/resp/benchmark_k_fix_time.py
   ```

4. Nosy in selecting top K models

   ```bash
   python ./internal/ml/model_selection/exps/micro/resp/benchmark_noisy_influence.py
   ```

5. Weight-sharing result

   ```bash
   nohup bash internal/ml/model_selection/scripts/benchmark_weight_sharing.sh &
   ```

# Run end2end model selection

download the dataset and put it in the `exp_data/data/structure_data`

```
python main.py --budget=100 --dataset=frappe
```

Check the log at the `logs_default`
