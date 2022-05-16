import calendar
import os
import time
from exps.shared_args import parse_arguments

args = parse_arguments()

# set the log name
gmt = time.gmtime()
ts = calendar.timegm(gmt)
os.environ.setdefault("log_logger_folder_name", f"bm_filter_phase")
os.environ.setdefault("log_file_name", f"bm_filter_{args.dataset}_{args.device}" + "_" + str(ts) + ".log")
os.environ.setdefault("base_dir", args.base_dir)

from src.eva_engine.run_ms import RunModelSelection
from src.common.constant import Config, CommonVars
import numpy as np
from src.tools.io_tools import write_json, read_json
from src.query_api.query_api_img import guess_score_time
from src.query_api.query_api_mlp import GTMLP


def debug_args(args, dataset):
    args.dataset = dataset
    args.base_dir = "../exp_data/"
    args.is_simulate = True
    args.log_folder = "log_ku_tradeoff"

    if dataset == Config.Frappe:
        args.search_space = "mlp_sp"
        args.epoch = 13
        args.hidden_choice_len = 20
        y_label = "FRAPPE"
        gtapi = GTMLP(dataset)
        score_time = gtapi.get_score_one_model_time("cpu")

    if dataset == Config.UCIDataset:
        args.search_space = "mlp_sp"
        args.epoch = 4
        args.hidden_choice_len = 20
        y_label = "DIABETES"
        gtapi = GTMLP(dataset)
        score_time = gtapi.get_score_one_model_time("cpu")

    if dataset == Config.Criteo:
        args.search_space = "mlp_sp"
        args.epoch = 9
        args.hidden_choice_len = 10
        y_label = "CRITEO"
        gtapi = GTMLP(dataset)
        score_time = gtapi.get_score_one_model_time("cpu")

    if dataset == Config.c10:
        args.search_space = "nasbench201"
        args.epoch = 200
        y_label = "C10"
        score_time = guess_score_time(Config.NB201, Config.c10)

    if dataset == Config.c100:
        args.search_space = "nasbench201"
        args.epoch = 200
        y_label = "C100"
        score_time = guess_score_time(Config.NB201, Config.c100)

    if dataset == Config.imgNet:
        args.search_space = "nasbench201"
        args.epoch = 200
        y_label = "IN-16"
        score_time = guess_score_time(Config.NB201, Config.imgNet)
    return y_label, score_time


def update_alg_name(name):
    if name == CommonVars.PRUNE_SYNFLOW:
        return "SynFlow"
    if name == CommonVars.JACFLOW:
        return "JacFlow"
    if name == CommonVars.PRUNE_SNIP:
        return "SNIP"

    return name


if __name__ == "__main__":

    # this is for ploting the graph

    for n in [500, 1000, 1500, 2000, 2500, 3000]:
        dataset_res = {}

        for dataset in [Config.UCIDataset, Config.Criteo, Config.Frappe, Config.c10, Config.c100, Config.imgNet, ]:
            y_label, score_time = debug_args(args, dataset)

            # this is for ploting the graph
            dataset_res[y_label] = []

            result = read_json(f"./internal/ml/model_selection/exp_result/combine_train_free_based_{dataset}_n_{n}.josn")

            rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)

            print("============" * 20)
            print(dataset)

            for training_free_alg, run_100_k_modesl in result.items():
                if training_free_alg == "nas_wot":
                    continue
                for budget_aware_alg in [Config.SUCCHALF, Config.SUCCREJCT, Config.UNIFORM]:
                    acc_result = []
                    time_result = []
                    for k_models in run_100_k_modesl:
                        best_arch, best_arch_performance, _, total_time_usage = rms.refinement_phase(
                            U=1, k_models=k_models,
                            alg_name=budget_aware_alg)
                        acc_result.append(best_arch_performance)
                        time_result.append(total_time_usage + score_time * len(k_models))

                    exp = np.array(acc_result)
                    q_75_y = np.quantile(exp, .75, axis=0)
                    q_25_y = np.quantile(exp, .25, axis=0)
                    mean_y = np.quantile(exp, .5, axis=0)

                    exp_time = np.array(time_result)
                    mean_time = np.quantile(exp_time, .5, axis=0)
                    # print(
                    #     f"Running task budget_aware_alg = {budget_aware_alg}, metrics = {training_free_alg}, "
                    #     f"0.25-0.75 = {q_25_y}, {mean_y}, {q_75_y},"
                    #     f" total_time_usage={mean_time}")

                    dataset_res[y_label].append(
                        [update_alg_name(training_free_alg) + " + " + budget_aware_alg, mean_y, mean_time]
                    )
        print(n, dataset_res)


"""
/Users/kevin/opt/anaconda3/envs/trails/bin/python /Users/kevin/project_python/VLDB_code/TRAILS/internal/ml/model_selection/exps/micro/benchmark_train_free_train_based_combines_phase2.py
local api running at ../exp_data/img_data
base_dir is ../exp_data/
local api running at ../exp_data/img_data
Loading ../exp_data/tab_data/uci_diabetes/all_train_baseline_uci_160k_40epoch.json...
Loading ../exp_data/tab_data/uci_diabetes/score_uci_diabetes_batch_size_32_all_metrics.json...
Loading ../exp_data/micro_sensitivity/3_batch_size/4/score_mlp_sp_uci_diabetes_batch_size_32_cpu.json...
Loading ./not_exist...
./not_exist is not exist
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_uci_diabetes_n_500.josn...
================================================================================================================================================================================================================================================
uci_diabetes
Loading ../exp_data/tab_data/criteo/all_train_baseline_criteo.json...
Loading ../exp_data/tab_data/criteo/score_criteo_batch_size_32.json...
Loading ../exp_data/micro_sensitivity/3_batch_size/4/score_mlp_sp_criteo_batch_size_32_cpu.json...
Loading ./not_exist...
./not_exist is not exist
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_criteo_n_500.josn...
================================================================================================================================================================================================================================================
criteo
Loading ../exp_data/tab_data/frappe/all_train_baseline_frappe.json...
Loading ../exp_data/tab_data/frappe/score_frappe_batch_size_32_local_finish_all_models.json...
Loading ../exp_data/micro_sensitivity/3_batch_size/4/score_mlp_sp_frappe_batch_size_32_cpu.json...
Loading ../exp_data/tab_data/weight_share_nas_frappe.json...
../exp_data/tab_data/weight_share_nas_frappe.json is not exist
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_frappe_n_500.josn...
================================================================================================================================================================================================================================================
frappe
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_cifar10_n_500.josn...
================================================================================================================================================================================================================================================
cifar10
Loading ../exp_data/img_data/ground_truth/201_allEpoch_info...
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_cifar100_n_500.josn...
================================================================================================================================================================================================================================================
cifar100
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_ImageNet16-120_n_500.josn...
================================================================================================================================================================================================================================================
ImageNet16-120
500 {'DIABETES': [['SynFlow + SUCCHALF', 0.626772872210564, 23.943343232403805], ['SynFlow + SUCCREJCT', 0.6254165259073317, 37.919051001797726], ['SynFlow + UNIFORM', 0.634490604355806, 59.48083872629266], ['SNIP + SUCCHALF', 0.6324450731694985, 24.261564086209347], ['SNIP + SUCCREJCT', 0.6324450731694985, 38.05694444013696], ['SNIP + UNIFORM', 0.639068923228679, 59.72796935869317], ['JacFlow + SUCCHALF', 0.626645822632548, 24.248920033703854], ['JacFlow + SUCCREJCT', 0.6241458161054811, 37.11021430326562], ['JacFlow + UNIFORM', 0.6341174344922629, 60.1202038033114]], 'CRITEO': [['SynFlow + SUCCHALF', 0.8029167026605589, 1557.2107951992728], ['SynFlow + SUCCREJCT', 0.8030174364188694, 2575.630772268174], ['SynFlow + UNIFORM', 0.803067534892734, 7708.149238860009], ['SNIP + SUCCHALF', 0.8030126932192898, 1469.5663378590323], ['SNIP + SUCCREJCT', 0.8028240889922882, 2285.3706115835885], ['SNIP + UNIFORM', 0.8030942983428183, 7234.808901464341], ['JacFlow + SUCCHALF', 0.8028094192773091, 1490.8040527456976], ['JacFlow + SUCCREJCT', 0.8028756807272186, 2319.1419192904214], ['JacFlow + UNIFORM', 0.8031232907610111, 7218.5816606634835]], 'FRAPPE': [['SynFlow + SUCCHALF', 0.9797083647863591, 79.85997582553864], ['SynFlow + SUCCREJCT', 0.9796469434148085, 124.31286300300599], ['SynFlow + UNIFORM', 0.9803075428282881, 557.1919834720993], ['SNIP + SUCCHALF', 0.979661243857286, 70.90328229545594], ['SNIP + SUCCREJCT', 0.979661243857286, 113.33963430999756], ['SNIP + UNIFORM', 0.9801938134634242, 495.35634649871827], ['JacFlow + SUCCHALF', 0.9796419176878983, 80.42643571971894], ['JacFlow + SUCCREJCT', 0.9796078149161596, 125.23744119285584], ['JacFlow + UNIFORM', 0.9803075428282881, 555.3384010898972]], 'C10': [['SynFlow + SUCCHALF', 94.36333333333334, 328.0454032291458], ['SynFlow + SUCCREJCT', 94.36333333333334, 516.1176318141165], ['SynFlow + UNIFORM', 94.36333333333334, 31661.66147314205], ['SNIP + SUCCHALF', 91.09, 225.22058509767623], ['SNIP + SUCCREJCT', 91.09, 354.09882258356186], ['SNIP + UNIFORM', 91.78999999999999, 21639.44888829173], ['JacFlow + SUCCHALF', 93.63, 343.87836347997757], ['JacFlow + SUCCREJCT', 93.63, 542.6464710503305], ['JacFlow + UNIFORM', 93.98333333333333, 33224.16474436704]], 'C100': [['SynFlow + SUCCHALF', 73.14666666666666, 327.6705024843216], ['SynFlow + SUCCREJCT', 73.14666666666666, 512.3382961414201], ['SynFlow + UNIFORM', 73.2, 31585.226593792908], ['SNIP + SUCCHALF', 66.19666666666666, 226.0558044489452], ['SNIP + SUCCREJCT', 66.19666666666666, 366.55609870288487], ['SNIP + UNIFORM', 66.19666666666666, 21688.004581409314], ['JacFlow + SUCCHALF', 70.70500000000001, 344.7356254519962], ['JacFlow + SUCCREJCT', 70.90666666666667, 544.5679262569064], ['JacFlow + UNIFORM', 71.215, 33283.529303020485]], 'IN-16': [['SynFlow + SUCCHALF', 46.35555552842882, 963.3779067760875], ['SynFlow + SUCCREJCT', 46.35555552842882, 1511.256884885629], ['SynFlow + UNIFORM', 46.35555552842882, 95572.5509532548], ['SNIP + SUCCHALF', 34.6666666692098, 639.4810724608444], ['SNIP + SUCCREJCT', 34.6666666692098, 1021.7187054721741], ['SNIP + UNIFORM', 35.64999999237061, 63355.356395110095], ['JacFlow + SUCCHALF', 45.13333332824707, 1009.2110850365616], ['JacFlow + SUCCREJCT', 45.40833334350586, 1602.1440973278059], ['JacFlow + UNIFORM', 45.45833334859212, 100158.7705527517]]}
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_uci_diabetes_n_1000.josn...
================================================================================================================================================================================================================================================
uci_diabetes
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_criteo_n_1000.josn...
================================================================================================================================================================================================================================================
criteo
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_frappe_n_1000.josn...
================================================================================================================================================================================================================================================
frappe
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_cifar10_n_1000.josn...
================================================================================================================================================================================================================================================
cifar10
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_cifar100_n_1000.josn...
================================================================================================================================================================================================================================================
cifar100
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_ImageNet16-120_n_1000.josn...
================================================================================================================================================================================================================================================
ImageNet16-120
1000 {'DIABETES': [['SynFlow + SUCCHALF', 0.6276538462943583, 73.34451784755908], ['SynFlow + SUCCREJCT', 0.6311991506508869, 126.35578038360796], ['SynFlow + UNIFORM', 0.6399547616709882, 118.46905281212054], ['SNIP + SUCCHALF', 0.6333132032321878, 76.72589864399157], ['SNIP + SUCCREJCT', 0.6373115809256313, 135.16413356926165], ['SNIP + UNIFORM', 0.6437914142608658, 117.61682321693621], ['JacFlow + SUCCHALF', 0.6276538462943583, 72.66465057995043], ['JacFlow + SUCCREJCT', 0.6290049481073127, 122.81854178573809], ['JacFlow + UNIFORM', 0.6399448020044406, 117.01054002907]], 'CRITEO': [['SynFlow + SUCCHALF', 0.8031232907610111, 5082.344331096407], ['SynFlow + SUCCREJCT', 0.8031991349785236, 9547.546375583406], ['SynFlow + UNIFORM', 0.8032278661585799, 15432.78462250113], ['SNIP + SUCCHALF', 0.8028094192773091, 4606.3050993192155], ['SNIP + SUCCREJCT', 0.8030191272016503, 8318.77351970553], ['SNIP + UNIFORM', 0.8032278661585799, 14361.843510459657], ['JacFlow + SUCCHALF', 0.8031232907610111, 4903.245997260805], ['JacFlow + SUCCREJCT', 0.8030174364188694, 9304.51239664435], ['JacFlow + UNIFORM', 0.8032278661585799, 14661.148913215395]], 'FRAPPE': [['SynFlow + SUCCHALF', 0.9796885305801462, 246.85283400772096], ['SynFlow + SUCCREJCT', 0.9798820501175385, 443.77895297763826], ['SynFlow + UNIFORM', 0.9807419022420424, 1119.2522309040833], ['SNIP + SUCCHALF', 0.9796885305801462, 219.37903465984346], ['SNIP + SUCCREJCT', 0.9796797028069436, 398.2868802761841], ['SNIP + UNIFORM', 0.9804345009402579, 986.0128888106156], ['JacFlow + SUCCHALF', 0.9796238796690948, 239.6017245268631], ['JacFlow + SUCCREJCT', 0.9795540413942015, 434.2294235443878], ['JacFlow + UNIFORM', 0.9805554781549316, 1080.8199963784027]], 'C10': [['SynFlow + SUCCHALF', 94.36333333333334, 1010.3483345610528], ['SynFlow + SUCCREJCT', 94.36333333333334, 1837.3993462170645], ['SynFlow + UNIFORM', 94.36333333333334, 62542.36254819406], ['SNIP + SUCCHALF', 91.09, 720.7629766282353], ['SNIP + SUCCREJCT', 91.09, 1341.9833965488615], ['SNIP + UNIFORM', 91.78999999999999, 43224.85476108277], ['JacFlow + SUCCHALF', 93.6725, 1076.6798184701147], ['JacFlow + SUCCREJCT', 93.475, 1972.9114427923928], ['JacFlow + UNIFORM', 93.98333333333333, 65922.52488108363]], 'C100': [['SynFlow + SUCCHALF', 73.14666666666666, 1003.0203713031269], ['SynFlow + SUCCREJCT', 73.14666666666666, 1815.9146246847877], ['SynFlow + UNIFORM', 73.50333333333333, 62247.333264313456], ['SNIP + SUCCHALF', 65.97000000000001, 723.1463541837193], ['SNIP + SUCCREJCT', 65.86500000000001, 1360.4672780996505], ['SNIP + UNIFORM', 66.64500000000001, 43136.20354063117], ['JacFlow + SUCCHALF', 71.215, 1084.0397882456098], ['JacFlow + SUCCREJCT', 70.92, 1976.9215736837841], ['JacFlow + UNIFORM', 71.215, 66291.8099733075]], 'IN-16': [['SynFlow + SUCCHALF', 46.35555552842882, 2963.888587220646], ['SynFlow + SUCCREJCT', 46.35555552842882, 5410.463900744653], ['SynFlow + UNIFORM', 46.40000000678168, 188550.7533471674], ['SNIP + SUCCHALF', 39.199999964396156, 2128.4173837442167], ['SNIP + SUCCREJCT', 39.199999964396156, 3961.5403401303183], ['SNIP + UNIFORM', 39.199999964396156, 128811.97523816819], ['JacFlow + SUCCHALF', 45.13333332824707, 3198.640402447678], ['JacFlow + SUCCREJCT', 45.13333332824707, 5845.766180991156], ['JacFlow + UNIFORM', 45.8722222120497, 199872.08666041621]]}
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_uci_diabetes_n_1500.josn...
================================================================================================================================================================================================================================================
uci_diabetes
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_criteo_n_1500.josn...
================================================================================================================================================================================================================================================
criteo
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_frappe_n_1500.josn...
================================================================================================================================================================================================================================================
frappe
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_cifar10_n_1500.josn...
================================================================================================================================================================================================================================================
cifar10
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_cifar100_n_1500.josn...
================================================================================================================================================================================================================================================
cifar100
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_ImageNet16-120_n_1500.josn...
================================================================================================================================================================================================================================================
ImageNet16-120
1500 {'DIABETES': [['SynFlow + SUCCHALF', 0.6276538462943583, 118.16491636732403], ['SynFlow + SUCCREJCT', 0.6356401788630915, 196.87187537649456], ['SynFlow + UNIFORM', 0.644274085739307, 180.46933850744549], ['SNIP + SUCCHALF', 0.6380491505445408, 118.99141273001018], ['SNIP + SUCCREJCT', 0.6434407281140964, 204.70431766012493], ['SNIP + UNIFORM', 0.6460524944239291, 177.70089969137493], ['JacFlow + SUCCHALF', 0.626772872210564, 114.76542565325084], ['JacFlow + SUCCREJCT', 0.6331101542270204, 191.93755969503704], ['JacFlow + UNIFORM', 0.644274085739307, 176.96764573553386]], 'CRITEO': [['SynFlow + SUCCHALF', 0.8031991349785236, 7862.208376989955], ['SynFlow + SUCCREJCT', 0.8031991349785236, 16569.683008537882], ['SynFlow + UNIFORM', 0.8032278661585799, 23354.937637911433], ['SNIP + SUCCHALF', 0.8030942983428183, 7293.57169757902], ['SNIP + SUCCREJCT', 0.8030942983428183, 15340.443044529551], ['SNIP + UNIFORM', 0.8032615745641593, 22017.51957582056], ['JacFlow + SUCCHALF', 0.8031991349785236, 7567.313978896731], ['JacFlow + SUCCREJCT', 0.8031991349785236, 16447.45051179945], ['JacFlow + UNIFORM', 0.8032278661585799, 22200.729629979724]], 'FRAPPE': [['SynFlow + SUCCHALF', 0.9796885305801462, 381.0430866801453], ['SynFlow + SUCCREJCT', 0.9800730107265379, 780.9159590327454], ['SynFlow + UNIFORM', 0.9808489735757652, 1665.2527841651154], ['SNIP + SUCCHALF', 0.9796885305801462, 344.4025401198578], ['SNIP + SUCCREJCT', 0.9796427849076285, 728.4791785800171], ['SNIP + UNIFORM', 0.9805225785303557, 1469.163978966446], ['JacFlow + SUCCHALF', 0.9796885305801462, 364.3281228863907], ['JacFlow + SUCCREJCT', 0.9800730107265379, 741.166241320343], ['JacFlow + UNIFORM', 0.9807657915745109, 1600.0643573128891]], 'C10': [['SynFlow + SUCCHALF', 94.36333333333334, 1568.1798831527346], ['SynFlow + SUCCREJCT', 94.37333333333333, 3208.0366774504296], ['SynFlow + UNIFORM', 94.37333333333333, 93149.3563511987], ['SNIP + SUCCHALF', 91.09, 1149.7003254568917], ['SNIP + SUCCREJCT', 91.09, 2418.8699376715704], ['SNIP + UNIFORM', 91.87, 66025.25967358032], ['JacFlow + SUCCHALF', 93.32499999999999, 1671.3951428381374], ['JacFlow + SUCCREJCT', 93.565, 3476.1799882158557], ['JacFlow + UNIFORM', 93.98333333333333, 98557.84006712175]], 'C100': [['SynFlow + SUCCHALF', 73.14666666666666, 1556.2759829636072], ['SynFlow + SUCCREJCT', 73.14666666666666, 3167.7777322520756], ['SynFlow + UNIFORM', 73.50333333333333, 92329.99443379656], ['SNIP + SUCCHALF', 65.09, 1141.4794955640978], ['SNIP + SUCCREJCT', 65.09, 2419.008556234405], ['SNIP + UNIFORM', 66.64500000000001, 65119.2991250147], ['JacFlow + SUCCHALF', 70.79500000000002, 1655.70987635751], ['JacFlow + SUCCREJCT', 71.215, 3386.855049631846], ['JacFlow + UNIFORM', 71.29666666666667, 98102.88287490798]], 'IN-16': [['SynFlow + SUCCHALF', 46.35555552842882, 4622.292179706403], ['SynFlow + SUCCREJCT', 46.35555552842882, 9558.720544635453], ['SynFlow + UNIFORM', 46.40000000678168, 280808.50203482027], ['SNIP + SUCCHALF', 39.199999964396156, 3357.7787680215947], ['SNIP + SUCCREJCT', 39.199999964396156, 7116.937786912244], ['SNIP + UNIFORM', 39.199999964396156, 196405.49294945004], ['JacFlow + SUCCHALF', 45.616666641235355, 4983.748305926254], ['JacFlow + SUCCREJCT', 45.40833334350586, 10309.091889280478], ['JacFlow + UNIFORM', 45.8722222120497, 299039.7144172684]]}
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_uci_diabetes_n_2000.josn...
================================================================================================================================================================================================================================================
uci_diabetes
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_criteo_n_2000.josn...
================================================================================================================================================================================================================================================
criteo
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_frappe_n_2000.josn...
================================================================================================================================================================================================================================================
frappe
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_cifar10_n_2000.josn...
================================================================================================================================================================================================================================================
cifar10
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_cifar100_n_2000.josn...
================================================================================================================================================================================================================================================
cifar100
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_ImageNet16-120_n_2000.josn...
================================================================================================================================================================================================================================================
ImageNet16-120
2000 {'DIABETES': [['SynFlow + SUCCHALF', 0.6324450731694985, 170.55821601681157], ['SynFlow + SUCCREJCT', 0.6384420326187368, 268.92957810692235], ['SynFlow + UNIFORM', 0.6452868634104227, 238.14363018803044], ['SNIP + SUCCHALF', 0.6485063872461523, 176.6764047556345], ['SNIP + SUCCREJCT', 0.6485063872461523, 283.2276444845621], ['SNIP + UNIFORM', 0.6496176698082966, 238.20240072540685], ['JacFlow + SUCCHALF', 0.6290049481073127, 169.522074740452], ['JacFlow + SUCCREJCT', 0.6399647213375358, 265.8252587728922], ['JacFlow + UNIFORM', 0.6460524944239291, 239.31027297310277]], 'CRITEO': [['SynFlow + SUCCHALF', 0.8031991349785236, 12620.483981630794], ['SynFlow + SUCCREJCT', 0.8031991349785236, 27859.316136739246], ['SynFlow + UNIFORM', 0.8032278661585799, 30986.747804782382], ['SNIP + SUCCHALF', 0.8030942983428183, 12019.973198792926], ['SNIP + SUCCREJCT', 0.8030942983428183, 28298.91093303918], ['SNIP + UNIFORM', 0.8033541930059981, 29242.33312668561], ['JacFlow + SUCCHALF', 0.8031991349785236, 12363.178739092342], ['JacFlow + SUCCREJCT', 0.8031991349785236, 26229.956693551532], ['JacFlow + UNIFORM', 0.8032278661585799, 29484.018676183216]], 'FRAPPE': [['SynFlow + SUCCHALF', 0.9802477277646887, 611.6404514741516], ['SynFlow + SUCCREJCT', 0.9806211047448514, 1331.6027179193115], ['SynFlow + UNIFORM', 0.980912834036346, 2211.8446325492478], ['SNIP + SUCCHALF', 0.9796797028069436, 567.3734722566223], ['SNIP + SUCCREJCT', 0.9798791095765752, 1276.2842140626526], ['SNIP + UNIFORM', 0.9805554781549316, 1955.3818341922379], ['JacFlow + SUCCHALF', 0.9800730107265379, 588.7458641957855], ['JacFlow + SUCCREJCT', 0.9800730107265379, 1309.2798982095337], ['JacFlow + UNIFORM', 0.9808489735757652, 2139.0945488643265]], 'C10': [['SynFlow + SUCCHALF', 94.21999999999998, 2576.963900316874], ['SynFlow + SUCCREJCT', 94.36333333333334, 5606.316646516869], ['SynFlow + UNIFORM', 94.37333333333333, 122962.40740177518], ['SNIP + SUCCHALF', 91.645, 1941.9595046758423], ['SNIP + SUCCREJCT', 91.09, 4277.319520556335], ['SNIP + UNIFORM', 91.87, 88336.9813679292], ['JacFlow + SUCCHALF', 93.32499999999999, 2786.8965536985847], ['JacFlow + SUCCREJCT', 93.565, 6173.181192665417], ['JacFlow + UNIFORM', 93.98333333333333, 130723.27969212248]], 'C100': [['SynFlow + SUCCHALF', 73.14666666666666, 2531.2397516583937], ['SynFlow + SUCCREJCT', 73.14666666666666, 5447.265930350464], ['SynFlow + UNIFORM', 73.50333333333333, 120430.11571405489], ['SNIP + SUCCHALF', 65.64333333333333, 1947.0927316448349], ['SNIP + SUCCREJCT', 65.97000000000001, 4356.175339266073], ['SNIP + UNIFORM', 66.64500000000001, 87814.25918661851], ['JacFlow + SUCCHALF', 71.07499999999999, 2650.908869892484], ['JacFlow + SUCCREJCT', 70.885, 5807.955749584221], ['JacFlow + UNIFORM', 71.805, 128686.29601675997]], 'IN-16': [['SynFlow + SUCCHALF', 46.40000000678168, 7631.924516833327], ['SynFlow + SUCCREJCT', 46.35555552842882, 16757.039823453953], ['SynFlow + UNIFORM', 46.40000000678168, 370280.3840952681], ['SNIP + SUCCHALF', 39.199999964396156, 5702.086446195501], ['SNIP + SUCCREJCT', 34.08333332824707, 12585.82275707387], ['SNIP + UNIFORM', 39.199999964396156, 266448.9964123927], ['JacFlow + SUCCHALF', 45.616666641235355, 8180.785473331757], ['JacFlow + SUCCREJCT', 45.09999997965495, 18173.854140884672], ['JacFlow + UNIFORM', 45.92499997965494, 395828.404416476]]}
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_uci_diabetes_n_2500.josn...
================================================================================================================================================================================================================================================
uci_diabetes
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_criteo_n_2500.josn...
================================================================================================================================================================================================================================================
criteo
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_frappe_n_2500.josn...
================================================================================================================================================================================================================================================
frappe
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_cifar10_n_2500.josn...
================================================================================================================================================================================================================================================
cifar10
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_cifar100_n_2500.josn...
================================================================================================================================================================================================================================================
cifar100
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_ImageNet16-120_n_2500.josn...
================================================================================================================================================================================================================================================
ImageNet16-120
2500 {'DIABETES': [['SynFlow + SUCCHALF', 0.6311991506508869, 215.09490536814238], ['SynFlow + SUCCREJCT', 0.6411755529206314, 340.61462210779695], ['SynFlow + UNIFORM', 0.644274085739307, 301.35495208864717], ['SNIP + SUCCHALF', 0.6496176698082966, 220.55512713556791], ['SNIP + SUCCREJCT', 0.6496176698082966, 350.2430144465497], ['SNIP + UNIFORM', 0.6502315774849694, 297.2983316100171], ['JacFlow + SUCCHALF', 0.6283888913130469, 211.66322623854185], ['JacFlow + SUCCREJCT', 0.6399647213375358, 333.05275737410096], ['JacFlow + UNIFORM', 0.6460524944239291, 299.53309415941743]], 'CRITEO': [['SynFlow + SUCCHALF', 0.8031991349785236, 15439.331309136738], ['SynFlow + SUCCREJCT', 0.8031991349785236, 37731.48511904179], ['SynFlow + UNIFORM', 0.8032278661585799, 39187.876774963726], ['SNIP + SUCCHALF', 0.8032615745641593, 14656.47943252026], ['SNIP + SUCCREJCT', 0.8030942983428183, 37777.3701503843], ['SNIP + UNIFORM', 0.8033541930059981, 36190.10849147736], ['JacFlow + SUCCHALF', 0.8031991349785236, 15098.015345272412], ['JacFlow + SUCCREJCT', 0.8031991349785236, 37362.07642656266], ['JacFlow + UNIFORM', 0.8032278661585799, 37366.724092301716]], 'FRAPPE': [['SynFlow + SUCCHALF', 0.9800730107265379, 729.5822055875778], ['SynFlow + SUCCREJCT', 0.9804016029598102, 1867.505370908928], ['SynFlow + UNIFORM', 0.980939578791173, 2756.382052355957], ['SNIP + SUCCHALF', 0.9800119420708356, 689.9753423273087], ['SNIP + SUCCREJCT', 0.9798838069598936, 1723.6076589166641], ['SNIP + UNIFORM', 0.9805554781549316, 2459.8149703084946], ['JacFlow + SUCCHALF', 0.9800730107265379, 731.7744777738571], ['JacFlow + SUCCREJCT', 0.9800730107265379, 1908.9950374900818], ['JacFlow + UNIFORM', 0.9808860892815189, 2718.3753132164]], 'C10': [['SynFlow + SUCCHALF', 94.21999999999998, 3095.682139770077], ['SynFlow + SUCCREJCT', 94.37333333333333, 7726.5558312466055], ['SynFlow + UNIFORM', 94.37333333333333, 150643.3438061009], ['SNIP + SUCCHALF', 91.645, 2383.854659195627], ['SNIP + SUCCREJCT', 91.39333333333333, 6116.7559137445405], ['SNIP + UNIFORM', 91.87, 111336.13781834944], ['JacFlow + SUCCHALF', 93.32499999999999, 3365.82818473112], ['JacFlow + SUCCREJCT', 93.565, 8402.48346273263], ['JacFlow + UNIFORM', 93.98333333333333, 162174.77560316544]], 'C100': [['SynFlow + SUCCHALF', 73.14666666666666, 3051.064588385991], ['SynFlow + SUCCREJCT', 71.74000000000001, 7406.5528339498405], ['SynFlow + UNIFORM', 73.50333333333333, 148328.66118654568], ['SNIP + SUCCHALF', 67.92, 2429.6642460180465], ['SNIP + SUCCREJCT', 67.92, 6272.88192891893], ['SNIP + UNIFORM', 67.92, 111666.17736265872], ['JacFlow + SUCCHALF', 71.07499999999999, 3218.260973201934], ['JacFlow + SUCCREJCT', 71.07499999999999, 8040.552407844746], ['JacFlow + UNIFORM', 71.805, 159101.87517362658]], 'IN-16': [['SynFlow + SUCCHALF', 45.199999974568684, 8957.858435308957], ['SynFlow + SUCCREJCT', 46.35555552842882, 22385.071163052915], ['SynFlow + UNIFORM', 46.40000000678168, 454234.7515625737], ['SNIP + SUCCHALF', 37.92499997202555, 6974.07125555839], ['SNIP + SUCCREJCT', 35.644444410536025, 17595.6761293136], ['SNIP + UNIFORM', 39.199999964396156, 336454.90624896635], ['JacFlow + SUCCHALF', 45.87499998982747, 9827.721037471816], ['JacFlow + SUCCREJCT', 45.45833334859212, 24595.165532202787], ['JacFlow + UNIFORM', 46.00555555555555, 489040.27938274713]]}
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_uci_diabetes_n_3000.josn...
================================================================================================================================================================================================================================================
uci_diabetes
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_criteo_n_3000.josn...
================================================================================================================================================================================================================================================
criteo
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_frappe_n_3000.josn...
================================================================================================================================================================================================================================================
frappe
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_cifar10_n_3000.josn...
================================================================================================================================================================================================================================================
cifar10
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_cifar100_n_3000.josn...
================================================================================================================================================================================================================================================
cifar100
Loading ./internal/ml/model_selection/exp_result/combine_train_free_based_ImageNet16-120_n_3000.josn...
================================================================================================================================================================================================================================================
ImageNet16-120
3000 {'DIABETES': [['SynFlow + SUCCHALF', 0.634490604355806, 269.0572549720443], ['SynFlow + SUCCREJCT', 0.6411755529206314, 414.48487454849845], ['SynFlow + UNIFORM', 0.6460524944239291, 360.9856669326461], ['SNIP + SUCCHALF', 0.6508997450735811, 278.7723997255004], ['SNIP + SUCCREJCT', 0.6508997450735811, 431.3942955156005], ['SNIP + UNIFORM', 0.6508997450735811, 356.8542292733825], ['JacFlow + SUCCHALF', 0.6324450731694985, 264.52919191319114], ['JacFlow + SUCCREJCT', 0.6411755529206314, 405.17304378945], ['JacFlow + UNIFORM', 0.6460524944239291, 359.4315535445846]], 'CRITEO': [['SynFlow + SUCCHALF', 0.8031991349785236, 20090.26545199989], ['SynFlow + SUCCREJCT', 0.8031991349785236, 45884.288601609456], ['SynFlow + UNIFORM', 0.8032278661585799, 47007.05946502327], ['SNIP + SUCCHALF', 0.8032615745641593, 18800.860603543508], ['SNIP + SUCCREJCT', 0.8031076134514598, 46743.97882923721], ['SNIP + UNIFORM', 0.8033541930059981, 43280.97272023796], ['JacFlow + SUCCHALF', 0.8031991349785236, 19249.200908991086], ['JacFlow + SUCCREJCT', 0.8031991349785236, 45532.77034971355], ['JacFlow + UNIFORM', 0.8032278661585799, 45662.77430626987]], 'FRAPPE': [['SynFlow + SUCCHALF', 0.9802477277646887, 947.2891667532349], ['SynFlow + SUCCREJCT', 0.9808489735757652, 2498.94722468132], ['SynFlow + UNIFORM', 0.980939578791173, 3344.0816952633286], ['SNIP + SUCCHALF', 0.9796797028069436, 883.7160179066086], ['SNIP + SUCCREJCT', 0.9800730107265379, 2288.876804654541], ['SNIP + UNIFORM', 0.9805554781549316, 2984.300577108803], ['JacFlow + SUCCHALF', 0.9802477277646887, 915.6449882435227], ['JacFlow + SUCCREJCT', 0.9808489735757652, 2501.226643388214], ['JacFlow + UNIFORM', 0.9809963581552738, 3236.2062850641632]], 'C10': [['SynFlow + SUCCHALF', 94.36333333333334, 3806.2317394789525], ['SynFlow + SUCCREJCT', 94.21999999999998, 10038.72972384571], ['SynFlow + UNIFORM', 94.37333333333333, 177797.85525320587], ['SNIP + SUCCHALF', 91.645, 3074.5808833303217], ['SNIP + SUCCREJCT', 90.72, 8197.145881003358], ['SNIP + UNIFORM', 91.87, 134000.00200711313], ['JacFlow + SUCCHALF', 93.49333333333334, 4189.191881055423], ['JacFlow + SUCCREJCT', 93.655, 10944.478628618874], ['JacFlow + UNIFORM', 93.98333333333333, 191748.99232730188]], 'C100': [['SynFlow + SUCCHALF', 73.14666666666666, 3802.4024396666355], ['SynFlow + SUCCREJCT', 73.14666666666666, 9837.557947983336], ['SynFlow + UNIFORM', 73.50333333333333, 176150.9954390994], ['SNIP + SUCCHALF', 67.92, 3123.163336082572], ['SNIP + SUCCREJCT', 67.92, 8616.321125264669], ['SNIP + UNIFORM', 67.92, 136171.18726489227], ['JacFlow + SUCCHALF', 71.07499999999999, 4088.1412497460724], ['JacFlow + SUCCREJCT', 71.07499999999999, 10748.939189935003], ['JacFlow + UNIFORM', 71.805, 189182.94485513796]], 'IN-16': [['SynFlow + SUCCHALF', 45.199999974568684, 11192.55959367173], ['SynFlow + SUCCREJCT', 46.35555552842882, 29356.86932018104], ['SynFlow + UNIFORM', 46.40000000678168, 536827.8685701462], ['SNIP + SUCCHALF', 36.64999997965495, 9097.982479796046], ['SNIP + SUCCREJCT', 34.08333332824707, 24112.95606710204], ['SNIP + UNIFORM', 39.199999964396156, 407362.6786587987], ['JacFlow + SUCCHALF', 45.616666641235355, 12277.664483765637], ['JacFlow + SUCCREJCT', 45.40833334350586, 32673.703014174895], ['JacFlow + UNIFORM', 46.48333334350586, 574630.5516308327]]}

Process finished with exit code 0

"""