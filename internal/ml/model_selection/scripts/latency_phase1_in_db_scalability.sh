

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection


#criteo
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency_sql.py \
  --embedding_cache_filtering=True \
  --tfmem=jacflow \
  --models_explore=500 \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=10 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cpu \
  --batch_size=64 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_cache_sql/

python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency_sql.py \
  --embedding_cache_filtering=True \
  --tfmem=jacflow \
  --models_explore=1000 \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=10 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cpu \
  --batch_size=64 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_cache_sql/


python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency_sql.py \
  --embedding_cache_filtering=True \
  --tfmem=jacflow \
  --models_explore=2000 \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=10 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cpu \
  --batch_size=64 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_cache_sql/


python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency_sql.py \
  --embedding_cache_filtering=True \
  --tfmem=jacflow \
  --models_explore=4000 \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=10 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cpu \
  --batch_size=64 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_cache_sql/

