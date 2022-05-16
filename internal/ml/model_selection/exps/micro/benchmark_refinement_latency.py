from shared_config import parse_config_arguments
from src.dataset_utils.stream_dataloader import StreamingDataLoader
from typing import Any, List, Dict, Tuple
from src.eva_engine.run_ms import RunModelSelection


def streaming_refinement_phase(u: int, k_models: List, dataset_name: str, config_file: str):
    """
    U: training-epoches
    K-Models: k models to train
    config_file: config file path
    """
    args = parse_config_arguments(config_file)
    args.device = "cuda:7"
    train_dataloader = StreamingDataLoader(
        cache_svc_url=args.cache_svc_url, table_name=f"{dataset_name}_train", name_space="train")
    eval_dataloader = StreamingDataLoader(
        cache_svc_url=args.cache_svc_url, table_name=f"{dataset_name}_valid", name_space="valid")

    try:
        rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
        best_arch, best_arch_performance, _ , _= rms.refinement_phase(
            U=u,
            k_models=k_models,
            train_loader=train_dataloader,
            valid_loader=eval_dataloader)
    finally:
        train_dataloader.stop()
        eval_dataloader.stop()
    return {"best_arch": best_arch, "best_arch_performance": best_arch_performance}


def sequence_refinement_phase(u: int, k_models: List, dataset_name: str, config_file: str):
    """
    U: training-epoches
    K-Models: k models to train
    config_file: config file path
    """
    args = parse_config_arguments(config_file)
    args.device = "cuda:7"
    train_dataloader = StreamingDataLoader(
        cache_svc_url=args.cache_svc_url, table_name=f"{dataset_name}_train", name_space="train")
    eval_dataloader = StreamingDataLoader(
        cache_svc_url=args.cache_svc_url, table_name=f"{dataset_name}_valid", name_space="valid")

    try:
        rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
        best_arch, best_arch_performance, _, _ = rms.refinement_phase(
            U=u,
            k_models=k_models,
            train_loader=train_dataloader,
            valid_loader=eval_dataloader)
    finally:
        train_dataloader.stop()
        eval_dataloader.stop()
    return {"best_arch": best_arch, "best_arch_performance": best_arch_performance}


def load_once_refinement_phase(u: int, k_models: List, dataset_name: str, config_file: str):
    """
    U: training-epoches
    K-Models: k models to train
    config_file: config file path
    """
    args = parse_config_arguments(config_file)
    args.device = "cuda:7"
    train_dataloader = StreamingDataLoader(
        cache_svc_url=args.cache_svc_url, table_name=f"{dataset_name}_train", name_space="train")
    eval_dataloader = StreamingDataLoader(
        cache_svc_url=args.cache_svc_url, table_name=f"{dataset_name}_valid", name_space="valid")

    try:
        rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
        best_arch, best_arch_performance, _ , _= rms.refinement_phase(
            U=u,
            k_models=k_models,
            train_loader=train_dataloader,
            valid_loader=eval_dataloader)
    finally:
        train_dataloader.stop()
        eval_dataloader.stop()
    return {"best_arch": best_arch, "best_arch_performance": best_arch_performance}









