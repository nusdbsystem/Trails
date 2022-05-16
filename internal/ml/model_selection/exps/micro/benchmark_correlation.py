import random

import numpy as np
import os
import numpy as np
from typing import List


# Initialize function to calculate correlation
def calculate_correlation(dataset, search_space, epoch_train, srcc_top_k: List = [1], is_visual=False):
    print("\n================================================================")
    print(f" {dataset} + {search_space}")
    print("================================================================")
    # Initialize query objects
    acc_query = SimulateTrain(space_name=search_space)
    score_query = SimulateScore(space_name=search_space, dataset_name=dataset)

    # Get list of all model IDs that have been trained and scored
    trained_models = acc_query.query_all_model_ids(dataset)
    scored_models = score_query.query_all_model_ids(dataset)

    # Find the intersection between trained_models and scored_models
    trained_scored_models = list(set(trained_models) & set(scored_models))

    # each alg have one corresponding ground truth list
    model_train_res_lst = {"nas_wot_add_syn_flow": []}
    all_alg_score_dic = {"nas_wot_add_syn_flow": []}

    # Populate algorithm scores and training results
    for model_id in trained_scored_models:
        score_value = score_query.query_all_tfmem_score(model_id)
        acc, _ = acc_query.get_ground_truth(arch_id=model_id, dataset=dataset, epoch_num=epoch_train)

        # append score and ground truth
        for alg, value in score_value.items():
            # If the algorithm is not in the dict, initialize its list
            if alg not in model_train_res_lst:
                model_train_res_lst[alg] = []
            if alg not in all_alg_score_dic:
                all_alg_score_dic[alg] = []

            model_train_res_lst[alg].append(acc)
            all_alg_score_dic[alg].append(float(value))

        if "nas_wot" in score_value:
            all_alg_score_dic["nas_wot_add_syn_flow"].append(
                float(score_value["nas_wot"]) + float(score_value["synflow"]))
            model_train_res_lst["nas_wot_add_syn_flow"].append(acc)
        else:
            all_alg_score_dic["nas_wot_add_syn_flow"].append(0)
            model_train_res_lst["nas_wot_add_syn_flow"].append(acc)

    # Measure the correlation for each algorithm and print the result
    for topp in srcc_top_k:
        print("--------------------------------------------------")
        for alg in all_alg_score_dic.keys():
            scores = all_alg_score_dic[alg]
            ground_truth = model_train_res_lst[alg]
            ground_truth_sorted_indices = np.argsort(ground_truth)[- int(topp * len(ground_truth)):]
            top_ground_truth = [ground_truth[i] for i in ground_truth_sorted_indices]
            top_scores = [scores[i] for i in ground_truth_sorted_indices]
            res = CorCoefficient.measure(top_scores, top_ground_truth)
            print(f"Top {topp, len(top_ground_truth)}, {alg}, {res[CommonVars.Spearman]}")

    # Get global ranks, measure correlation and print result for JACFLOW
    # if dataset in [Config.Frappe, Config.UCIDataset, Config.Criteo]:
    try:
        global_rank_score = [score_query.query_tfmem_rank_score(model_id)['nas_wot_synflow']
                             for model_id in trained_scored_models]

        sorted_indices = np.argsort(global_rank_score)
        # here the ground truth list should be same as nas_wot.., since they share the same model list
        sorted_ground_truth = [model_train_res_lst['nas_wot'][i] for i in sorted_indices]
        sorted_scores = [global_rank_score[i] for i in sorted_indices]

        res = CorCoefficient.measure(sorted_scores, sorted_ground_truth)
        print("JACFLOW", res[CommonVars.Spearman])
    except Exception as e:
        print("JACFLOW not provided, ", e)

    if is_visual:
        for alg in all_alg_score_dic.keys():
            scores = all_alg_score_dic[alg]
            ground_truth = model_train_res_lst[alg]
            ground_truth_sorted_indices = np.argsort(ground_truth)
            top_ground_truth = [ground_truth[i] for i in ground_truth_sorted_indices]
            top_scores = [scores[i] for i in ground_truth_sorted_indices]
            visualization(alg, dataset, top_ground_truth, top_scores)


def visualization(alg, dataset, sorted_ground_truth, sorted_scores):

    import matplotlib
    import matplotlib.pyplot as plt
    set_tick_size = 20
    matplotlib.rc('xtick', labelsize=set_tick_size)
    matplotlib.rc('ytick', labelsize=set_tick_size)
    plt.rcParams['axes.labelsize'] = set_tick_size
    color_list = ['blue', 'green', 'purple',  'black',  'brown', 'red']
    fig, ax = plt.subplots(figsize=(6.4, 5))
    plt.scatter(sorted_ground_truth, sorted_scores, color=random.choice(color_list))
    plt.grid()
    plt.xlabel('Validation AUC')
    plt.ylabel('TRAILER Score')
    plt.tight_layout()
    fig.savefig(f"visua_score_auc_{alg}_{dataset}.jpg", bbox_inches='tight')


# Call the main function
if __name__ == "__main__":
    os.environ.setdefault("base_dir", "../exp_data")
    from src.query_api.interface import SimulateTrain, SimulateScore
    from src.tools.correlation import CorCoefficient
    from src.common.constant import CommonVars, Config

    # Frappe configuration, here also measure SRCC of top 0.2% -> 0.8%
    calculate_correlation(Config.Frappe, Config.MLPSP, 13, srcc_top_k=[1], is_visual=False)

    # UCI configuration
    calculate_correlation(Config.UCIDataset, Config.MLPSP, 0, is_visual=False)

    # Criteo configuration
    calculate_correlation(Config.Criteo, Config.MLPSP, 9, is_visual=False)

    # NB101 + C10
    calculate_correlation(Config.c10, Config.NB101, None)

    # NB201 + C10
    calculate_correlation(Config.c10, Config.NB201, None)

    # NB201 + C100
    calculate_correlation(Config.c100, Config.NB201, None)

    # NB201 + imageNet
    calculate_correlation(Config.imgNet, Config.NB201, None)

    """
    ================================================================
     frappe + mlp_sp
    ================================================================
    Loading ../exp_data/tab_data/frappe/all_train_baseline_frappe.json...
    Loading ../exp_data/tab_data/frappe/score_frappe_batch_size_32_local_finish_all_models.json...
    Loading ../exp_data/tab_data/expressflow_score_mlp_sp_frappe_batch_size_32_cpu.json...
    Loading ../exp_data/tab_data/weight_share_nas_frappe.json...
    --------------------------------------------------
    Top (0.005, 800), nas_wot_add_syn_flow, 0.1986753573052458
    Top (0.005, 800), grad_norm, 0.12076418883266768
    Top (0.005, 800), grad_plain, 0.006358620227609798
    Top (0.005, 800), nas_wot, 0.19057972309084753
    Top (0.005, 800), ntk_cond_num, -0.0466050197351888
    Top (0.005, 800), ntk_trace, 0.08758335876257788
    Top (0.005, 800), ntk_trace_approx, 0.05574130182201712
    Top (0.005, 800), fisher, 0.0641524012888946
    Top (0.005, 800), grasp, -0.018840689722475004
    Top (0.005, 800), snip, 0.19218164902774643
    Top (0.005, 800), synflow, 0.1986753573052458
    Top (0.005, 800), weight_norm, 0.20104446565490514
    Top (0.005, 800), express_flow, 0.22902225628477543
    Top (0.005, 10), weight_share, 0.43030303030303024
    --------------------------------------------------
    Top (1, 160000), nas_wot_add_syn_flow, 0.7722624738302497
    Top (1, 160000), grad_norm, 0.5022881504556264
    Top (1, 160000), grad_plain, 0.04248195958085427
    Top (1, 160000), nas_wot, 0.6378172532674963
    Top (1, 160000), ntk_cond_num, -0.7226973187313154
    Top (1, 160000), ntk_trace, 0.5339715485407944
    Top (1, 160000), ntk_trace_approx, 0.15199042944834584
    Top (1, 160000), fisher, 0.503764400646809
    Top (1, 160000), grasp, -0.32104009686859214
    Top (1, 160000), snip, 0.7038939006969223
    Top (1, 160000), synflow, 0.7721962598662835
    Top (1, 160000), weight_norm, 0.7473390604383027
    Top (1, 160000), express_flow, 0.8028456677612497
    Top (1, 2000), weight_share, 0.517565395891349
    JACFLOW 0.7406547545980877
    
    ================================================================
     uci_diabetes + mlp_sp
    ================================================================
    Loading ../exp_data/tab_data/uci_diabetes/all_train_baseline_uci_160k_40epoch.json...
    Loading ../exp_data/tab_data/uci_diabetes/score_uci_diabetes_batch_size_32_all_metrics.json...
    Loading ../exp_data/tab_data/expressflow_score_mlp_sp_uci_diabetes_batch_size_32_cpu.json...
    Loading ./not_exist...
    ./not_exist is not exist
    --------------------------------------------------
    Top (1, 160000), nas_wot_add_syn_flow, 0.6855057216575722
    Top (1, 160000), grad_norm, 0.3999081089630585
    Top (1, 160000), grad_plain, 0.02451448778377102
    Top (1, 160000), nas_wot, 0.635540008950723
    Top (1, 160000), ntk_cond_num, -0.5654103067100021
    Top (1, 160000), ntk_trace, 0.3774899968561059
    Top (1, 160000), ntk_trace_approx, 0.31808993358325754
    Top (1, 160000), fisher, 0.21598774748021798
    Top (1, 160000), grasp, -0.23202305383871977
    Top (1, 160000), snip, 0.629837846386711
    Top (1, 160000), synflow, 0.6855051126181101
    Top (1, 160000), weight_norm, 0.6927936726919207
    Top (1, 160000), express_flow, 0.6978445139608305
    JACFLOW 0.692050239116883
    
    ================================================================
     criteo + mlp_sp
    ================================================================
    Loading ../exp_data/tab_data/criteo/all_train_baseline_criteo.json...
    Loading ../exp_data/tab_data/criteo/score_criteo_batch_size_32.json...
    Loading ../exp_data/tab_data/expressflow_score_mlp_sp_criteo_batch_size_32_cpu.json...
    Loading ./not_exist...
    ./not_exist is not exist
    --------------------------------------------------
    Top (1, 10000), nas_wot_add_syn_flow, 0.7464429461404294
    Top (1, 10000), grad_norm, 0.3471953123725521
    Top (1, 10000), grad_plain, 0.023434944830985988
    Top (1, 10000), nas_wot, 0.7128521207543811
    Top (1, 10000), ntk_cond_num, -0.6335174238677821
    Top (1, 10000), ntk_trace, 0.49024945003576803
    Top (1, 10000), ntk_trace_approx, 0.00890247410055012
    Top (1, 10000), fisher, 0.4302424148655023
    Top (1, 10000), grasp, -0.2026179640580912
    Top (1, 10000), snip, 0.7978576087791914
    Top (1, 10000), synflow, 0.7464395938803958
    Top (1, 10000), weight_norm, 0.8134301266060824
    Top (1, 10000), express_flow, 0.8276736303927363
    JACFLOW 0.7602146996144342
    
    ================================================================
     cifar10 + nasbench101
    ================================================================
    Loading ../exp_data/img_data/score_101_15k_c10_128.json...
    Loading ../exp_data/img_data/expssflow_score_nasbench101_cifar10_batch_size_32_cpu.json...
    Loading pickel ../exp_data/img_data/ground_truth/nasbench1_accuracy.p...
    Loading ../exp_data/img_data/ground_truth/nb101_id_to_hash.json...
    Loading ../exp_data/img_data/ground_truth/101_allEpoch_info_json...
    --------------------------------------------------
    Top (1, 15625), nas_wot_add_syn_flow, 0.3689520399131206
    Top (1, 15625), grad_norm, -0.24028712904763305
    Top (1, 15625), grad_plain, -0.37437473256641196
    Top (1, 15625), jacob_conv, -0.004427148034070742
    Top (1, 15625), nas_wot, 0.36881036232313413
    Top (1, 15625), ntk_cond_num, -0.30221532514959765
    Top (1, 15625), ntk_trace, -0.30751131315805114
    Top (1, 15625), ntk_trace_approx, -0.4195178764767932
    Top (1, 15625), fisher, -0.27330982397300113
    Top (1, 15625), grasp, 0.27891820857555477
    Top (1, 15625), snip, -0.1668017404578911
    Top (1, 15625), synflow, 0.3689520399131206
    Top (1, 15625), weight_norm, 0.5332072603621669
    Top (1, 73), express_flow, 0.3754348914881334
    JACFLOW 0.412358290561836
    
    ================================================================
     cifar10 + nasbench201
    ================================================================
    Loading ../exp_data/img_data/score_201_15k_c10_bs32_ic16.json...
    Loading ../exp_data/img_data/expssflow_score_nasbench201_cifar10_batch_size_32_cpu.json...
    Loading ../exp_data/img_data/ground_truth/201_allEpoch_info...
    --------------------------------------------------
    Top (1, 15625), nas_wot_add_syn_flow, 0.778881359475867
    Top (1, 15625), grad_norm, 0.6407726431247389
    Top (1, 15625), grad_plain, -0.12987923450265587
    Top (1, 15625), nas_wot, 0.7932888939510635
    Top (1, 15625), ntk_cond_num, -0.48387478988448707
    Top (1, 15625), ntk_trace, 0.37783839330319213
    Top (1, 15625), ntk_trace_approx, 0.346026013974282
    Top (1, 15625), fisher, 0.3880671330868219
    Top (1, 15625), grasp, 0.5301491874432275
    Top (1, 15625), snip, 0.6437364743868734
    Top (1, 15625), weight_norm, 0.005565168122791671
    Top (1, 15625), synflow, 0.7788685936507451
    Top (1, 2001), express_flow, 0.7808596435404074
    JACFLOW 0.8339798847659702
    
    ================================================================
     cifar100 + nasbench201
    ================================================================
    Loading ../exp_data/img_data/score_201_15k_c100_bs32_ic16.json...
    Loading ../exp_data/img_data/expssflow_score_nasbench201_cifar100_batch_size_32_cpu.json...
    --------------------------------------------------
    Top (1, 15000), nas_wot_add_syn_flow, 0.7671956422900841
    Top (1, 15000), grad_norm, 0.638398024328767
    Top (1, 15000), grad_plain, -0.16701447428313634
    Top (1, 15000), nas_wot, 0.8089325143676851
    Top (1, 15000), ntk_cond_num, -0.39182378815354696
    Top (1, 15000), ntk_trace, 0.37724922855703374
    Top (1, 15000), ntk_trace_approx, 0.38385292527377407
    Top (1, 15000), fisher, 0.3845332624562634
    Top (1, 15000), grasp, 0.5462288460061152
    Top (1, 15000), snip, 0.6375851983100865
    Top (1, 15000), weight_norm, 0.011918450096024165
    Top (1, 15000), synflow, 0.7671950896881894
    Top (1, 1920), express_flow, 0.7662904274045729
    JACFLOW 0.836747529703176
    
    ================================================================
     ImageNet16-120 + nasbench201
    ================================================================
    Loading ../exp_data/img_data/score_201_15k_imgNet_bs32_ic16.json...
    Loading ../exp_data/img_data/expssflow_score_nasbench201_ImageNet16-120_batch_size_32_cpu.json...
    --------------------------------------------------
    Top (1, 15000), nas_wot_add_syn_flow, 0.7450159967363084
    Top (1, 15000), grad_norm, 0.566172650250217
    Top (1, 15000), grad_plain, -0.16454617540967373
    Top (1, 15000), nas_wot, 0.7769715502067605
    Top (1, 15000), ntk_cond_num, -0.41263954976382056
    Top (1, 15000), ntk_trace, 0.310570269337782
    Top (1, 15000), ntk_trace_ap![](../../../../../visua_score_auc_express_flow_frappe.jpg)prox, 0.3566322129734418
    Top (1, 15000), fisher, 0.3202230462329743
    Top (1, 15000), grasp, 0.5093070840387243
    Top (1, 15000), snip, 0.5688946225005688
    Top (1, 15000), weight_norm, 0.005571648346911519
    Top (1, 15000), synflow, 0.7450170886565295
    Top (1, 1920), express_flow, 0.743078455158988
    JACFLOW 0.8077842182522329
"""
