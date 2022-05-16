time_16 = open('./internal/ml/model_selection/exp_result//score_imageNet_16x16', 'r')
lines16 = time_16.readlines()

time_32 = open('./internal/ml/model_selection/exp_result//score_imageNet_224x224', 'r')
lines32 = time_32.readlines()

for ele in [lines16, lines32]:
    time16_sum = 0
    synflow_time = 0
    nas_wot_time = 0
    total_modles = 0
    for ele2 in ele:
        time16_sum += float(ele2.split(" ")[0].strip()) + float(ele2.split(" ")[1].strip())
        synflow_time += float(ele2.split(" ")[0].strip())
        nas_wot_time += float(ele2.split(" ")[1].strip())
        total_modles += 1

    print(f"total_modles={total_modles}, time16_sum={time16_sum / total_modles}, "
          f"synflow_time={synflow_time / total_modles}, nas_wot_time={nas_wot_time / total_modles}")
