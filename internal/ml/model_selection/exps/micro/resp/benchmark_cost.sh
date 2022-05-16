

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
conda activate trails



# here all are on frappe dataset

# local try
# python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
#  --tfmem=grad_norm \
#  --models_explore=5000 \
#  --log_name=grad_norm \
#  --search_space=mlp_sp \
#  --num_layers=4 \
#  --hidden_choice_len=20 \
#  --base_dir=../exp_data/ \
#  --num_labels=2 \
#  --device=cpu \
#  --batch_size=32 \
#  --dataset=frappe \
#  --nfeat=5500 \
#  --nfield=10 \
#  --nemb=10 \
#  --workers=0 \
#  --result_dir=./internal/ml/model_selection/exp_result/ \
#  --log_folder=log_score_time_frappe

echo "Begin"
# grad_norm
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --tfmem=grad_norm \
  --models_explore=5000 \
  --log_name=grad_norm \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_frappe


# nas_wot
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --tfmem=nas_wot \
  --models_explore=5000 \
  --log_name=nas_wot \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_frappe


# ntk_cond_num
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --tfmem=ntk_cond_num \
  --models_explore=5000 \
  --log_name=ntk_cond_num \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_frappe

# ntk_trace
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --tfmem=ntk_trace \
  --models_explore=5000 \
  --log_name=ntk_trace \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_frappe

# ntk_trace_approx
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --tfmem=ntk_trace_approx \
  --models_explore=5000 \
  --log_name=ntk_trace_approx \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_frappe

# fisher
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --tfmem=fisher \
  --models_explore=5000 \
  --log_name=fisher \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_frappe

# grasp
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --tfmem=grasp \
  --models_explore=5000 \
  --log_name=grasp \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_frappe

# snip
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --tfmem=snip \
  --models_explore=5000 \
  --log_name=snip \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_frappe

# synflow
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --tfmem=synflow \
  --models_explore=5000 \
  --log_name=synflow \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_frappe

# express_flow
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --tfmem=express_flow \
  --models_explore=5000 \
  --log_name=express_flow \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_frappe



echo "Done"