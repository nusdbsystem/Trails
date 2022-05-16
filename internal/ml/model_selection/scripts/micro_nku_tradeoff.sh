

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection


# try to draw using the exsiting results
# this is image
python internal/ml/model_selection/exps/micro/draw_benchmark_nk_img.py
# this is the tabular
python internal/ml/model_selection/exps/micro/draw_benchmark_nk.py

# ====================================
# ====================================
# determine the K and U tradeoff
# ====================================
# ====================================
# frappe
python internal/ml/model_selection/exps/micro/benchmark_ku.py \
  --tfmem jacflow \
  --search_space mlp_sp \
  --epoch 14 \
  --hidden_choice_len 20 \
  --dataset frappe \
  --base_dir ../exp_data/ \
  --is_simulate True \
  --log_folder log_ku_tradeoff


# uci
python internal/ml/model_selection/exps/micro/benchmark_ku.py \
  --tfmem jacflow \
  --search_space mlp_sp \
  --hidden_choice_len 20 \
  --epoch 1 \
  --dataset uci_diabetes \
  --base_dir ../exp_data/ \
  --is_simulate True \
  --log_folder log_ku_tradeoff

# criteo
python internal/ml/model_selection/exps/micro/benchmark_ku.py \
  --tfmem jacflow \
  --search_space mlp_sp \
  --hidden_choice_len 10 \
  --epoch 10 \
  --dataset criteo \
  --base_dir ../exp_data/ \
  --is_simulate True \
  --log_folder log_ku_tradeoff


# c10
python internal/ml/model_selection/exps/micro/benchmark_ku.py \
  --tfmem jacflow \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset cifar10 \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff


# c100
python internal/ml/model_selection/exps/micro/benchmark_ku.py \
  --tfmem jacflow \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset cifar100 \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff


# imageNet
python internal/ml/model_selection/exps/micro/benchmark_ku.py \
  --tfmem jacflow \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset ImageNet16-120 \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff



# ====================================
# ====================================
# determine the K and n tradeoff
# ====================================
# ====================================


python internal/ml/model_selection/exps/micro/benchmark_nk.py \
  --tfmem jacflow \
  --search_space mlp_sp \
  --epoch 14 \
  --hidden_choice_len 20 \
  --dataset frappe \
  --base_dir ../exp_data/ \
  --is_simulate True \
  --log_folder log_ku_tradeoff


#uci
python internal/ml/model_selection/exps/micro/benchmark_nk.py \
  --tfmem jacflow \
  --search_space mlp_sp \
  --hidden_choice_len 20 \
  --epoch 1 \
  --dataset uci_diabetes \
  --base_dir ../exp_data/ \
  --is_simulate True \
  --log_folder log_ku_tradeoff


# criteo
python internal/ml/model_selection/exps/micro/benchmark_nk.py \
  --tfmem jacflow \
  --search_space mlp_sp \
  --hidden_choice_len 10 \
  --epoch 10 \
  --dataset criteo \
  --base_dir ../exp_data/ \
  --is_simulate True \
  --log_folder log_ku_tradeoff



# c10
python internal/ml/model_selection/exps/micro/benchmark_nk.py \
  --tfmem jacflow \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset cifar10 \
  --base_dir ../exp_data/ \
  --is_simulate True \
  --log_folder log_ku_tradeoff


# c100
python internal/ml/model_selection/exps/micro/benchmark_nk.py \
  --tfmem jacflow \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset cifar100 \
  --base_dir ../exp_data/ \
  --is_simulate True \
  --log_folder log_ku_tradeoff


# imageNet
python internal/ml/model_selection/exps/micro/benchmark_nk.py \
  --tfmem jacflow \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset ImageNet16-120 \
  --base_dir ../exp_data/ \
  --is_simulate True \
  --log_folder log_ku_tradeoff

