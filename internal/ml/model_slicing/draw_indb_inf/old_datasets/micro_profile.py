# Collecting data for plotting

datasets_result_10k = {'data_query_time': 0.5223848819732666, 'py_conver_to_tensor': 0.05331301689147949,
                       'tensor_to_gpu': 0.00040650367736816406, 'py_compute': 0.8352866172790527,
                       'load_model': 0.16507482528686523, 'overall_query_latency': 1.5977146625518799}
datasets_result_20k = {'data_query_time': 0.9279088973999023, 'py_conver_to_tensor': 0.08997559547424316,
                       'tensor_to_gpu': 0.0001285076141357422, 'py_compute': 1.0500628566741943,
                       'load_model': 0.15180516242980957, 'overall_query_latency': 3.1019253730773926}
datasets_result_40k = {'data_query_time': 1.8355934619903564, 'py_conver_to_tensor': 0.07959046363830566,
                       'tensor_to_gpu': 0.0002346038818359375, 'py_compute': 2.173098516464233,
                       'load_model': 0.1675722599029541, 'overall_query_latency': 7.379018783569336}

for datasets_result_used in [datasets_result_10k, datasets_result_20k, datasets_result_40k]:
    print("===" * 10)
    total_usage = datasets_result_used["data_query_time"] + \
                  datasets_result_used["py_conver_to_tensor"] + \
                  datasets_result_used["tensor_to_gpu"] + \
                  datasets_result_used["py_compute"] + \
                  datasets_result_used["load_model"]

    print(total_usage - datasets_result_used["overall_query_latency"])

    print(f"load_model = "
          f"{100 * datasets_result_used['load_model'] / total_usage}")
    print(f"data_query_time = "
          f"{100 * datasets_result_used['data_query_time'] / total_usage}")
    print(f"py_conver_to_tensor = "
          f"{100 * datasets_result_used['py_conver_to_tensor'] / total_usage}")
    print(f"py_compute & compute = "
          f"{100 * (datasets_result_used['py_compute'] + datasets_result_used['py_conver_to_tensor']) / total_usage}")
