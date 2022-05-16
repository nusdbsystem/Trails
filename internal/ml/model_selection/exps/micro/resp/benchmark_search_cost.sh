

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
conda activate trails


# grad_norm
# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=grad_norm \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10

# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=grad_norm \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10


# nas_wot
# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=nas_wot \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10

# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=nas_wot \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10



# ntk_cond_num
# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=ntk_cond_num \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10

# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=ntk_cond_num \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10





# ntk_trace
# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=ntk_trace \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10

# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=ntk_trace \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10





# ntk_trace_approx
# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=ntk_trace_approx \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10

# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=ntk_trace_approx \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10





# fisher
# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=fisher \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10

# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=fisher \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10




# grasp
# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=grasp \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10

# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=grasp \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10





# snip
# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=snip \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10

# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=snip \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10




# synflow
# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=synflow \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10

# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=synflow \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10


# express_flow
# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=express_flow \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=../exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10

# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --tfmem=express_flow \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
      --base_dir=../exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 10


# draw the plot
python ./internal/ml/model_selection/exps/micro/resp/search_cost_draw.py