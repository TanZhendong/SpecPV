import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from datasets import load_dataset

from specpv import SpecConfig, Speculator
from specpv.kv.kv_cache import initialize_past_key_values
from specpv.speculate.utils import (
    chunked_prefilling,
    evaluate_posterior,
    prepare_logits_processor,
    reset_tree_mode,
    should_partial_verify,
    tree_decoding,
    update_inference_inputs,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/pg19_test/pg19_test_60k.parquet",
    )
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/home/lthpc/nvmessd/zhendong/models/LLAMA3.1-8B-Instruct",
    )
    parser.add_argument(
        "--draft_model_path",
        type=str,
        default="/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K",
    )
    parser.add_argument("--context_length", type=int, default=32768)
    parser.add_argument(
        "--partial_length",
        type=int,
        default=8192,
        help=(
            "Legacy SpecPV retrieval-token budget. Used when --kv_budget is not set; "
            "it reproduces the "
            "previous test configuration: partial_length retrieval tokens plus "
            "32 sink and 128 window tokens."
        ),
    )
    parser.add_argument(
        "--kv_selector",
        choices=["sliding", "streaming", "h2o", "quest", "specpv"],
        default="specpv",
        help="KV selector for the shadow verifier.",
    )
    parser.add_argument(
        "--kv_budget",
        type=int,
        default=None,
        help="Persistent KV-token budget for a selector experiment (overrides legacy --partial_length).",
    )
    parser.add_argument(
        "--sink_tokens",
        type=int,
        default=32,
        help="Number of sink tokens for streaming, H2O, and SpecPV.",
    )
    parser.add_argument(
        "--window_tokens",
        type=int,
        default=128,
        help="Recent-token window for H2O and Quest.",
    )
    parser.add_argument("--partial_spec_tokens", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=65000)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def iter_progress(items, desc, position=0):
    if tqdm is not None:
        return tqdm(items, desc=desc, position=position)
    print(f"{desc}: {len(items)} samples")
    return items


def build_chat(tokenizer, text, model_name):
    system_prompt = (
        "You are a creative writing assistant. Continue the following story "
        "in a coherent, engaging, and stylistically consistent way."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    if "qwen3" in model_name:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_selector_config(args):
    """Allocate exactly ``kv_budget`` persistent slots for one selector."""
    block_size = SpecConfig.block_size
    if args.sink_tokens % block_size or args.window_tokens % block_size:
        raise ValueError(f"--sink_tokens and --window_tokens must be multiples of {block_size}")
    if args.kv_budget is None:
        if args.kv_selector != "specpv":
            raise ValueError("--partial_length is the legacy SpecPV-only setting; use --kv_budget for other selectors")
        if args.partial_length <= 0 or args.partial_length % block_size:
            raise ValueError(f"--partial_length must be a positive multiple of {block_size}")
        return SpecConfig(
            enable_offload=False,
            enable_partial_kv=True,
            kv_selector="quest",  # internal query-aware scorer
            n_sink_blocks=args.sink_tokens // block_size,
            n_retrieval_blocks=args.partial_length // block_size,
            n_window_blocks=args.window_tokens // block_size,
            partial_spec_tokens=args.partial_spec_tokens,
        )

    if args.kv_budget <= 0 or args.kv_budget % block_size:
        raise ValueError(f"--kv_budget must be a positive multiple of {block_size}")
    budget_blocks = args.kv_budget // block_size
    sink_blocks = args.sink_tokens // block_size
    window_blocks = args.window_tokens // block_size
    if args.kv_selector == "sliding":
        sink_blocks, retrieval_blocks, window_blocks = 0, 0, budget_blocks
    elif args.kv_selector == "streaming":
        retrieval_blocks = 0
        window_blocks = budget_blocks - sink_blocks
    elif args.kv_selector == "quest":
        # QUEST baseline: query-aware retrieval plus a local window, but no
        # SpecPV sink tokens.
        sink_blocks = 0
        retrieval_blocks = budget_blocks - window_blocks
    else:  # query-free H2O baseline and query-aware SpecPV
        retrieval_blocks = budget_blocks - sink_blocks - window_blocks

    if min(sink_blocks, retrieval_blocks, window_blocks) < 0:
        raise ValueError(
            f"KV budget {args.kv_budget} is too small for {args.kv_selector}: "
            f"sink={args.sink_tokens}, window={args.window_tokens}"
        )
    return SpecConfig(
        enable_offload=False,
        enable_partial_kv=True,
        # Both QUEST and SpecPV use the existing query-aware scorer. Their
        # cache layouts differ only in whether sink tokens are retained.
        kv_selector="quest" if args.kv_selector in {"quest", "specpv"} else args.kv_selector,
        n_sink_blocks=sink_blocks,
        n_retrieval_blocks=retrieval_blocks,
        n_window_blocks=window_blocks,
        partial_spec_tokens=args.partial_spec_tokens,
    )


def accepted_hidden_states(hidden_state_new, retrieve_indices, best_candidate, accept_length):
    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    return retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]


def full_reference_tree_decoding(
    model,
    tree_candidates,
    full_past_key_values,
    partial_past_key_values,
    tree_position_ids,
    input_ids,
    retrieve_indices,
    refresh_partial_cache,
):
    full_past_key_values.enabled = True
    if refresh_partial_cache:
        return tree_decoding(
            model,
            tree_candidates,
            full_past_key_values,
            partial_past_key_values,
            tree_position_ids,
            input_ids,
            retrieve_indices,
        )

    prev_partial_enabled = partial_past_key_values.enabled
    partial_past_key_values.enabled = False
    try:
        return tree_decoding(
            model,
            tree_candidates,
            full_past_key_values=full_past_key_values,
            partial_past_key_values=partial_past_key_values,
            tree_position_ids=tree_position_ids,
            input_ids=input_ids,
            retrieve_indices=retrieve_indices,
        )
    finally:
        partial_past_key_values.enabled = prev_partial_enabled


def shadow_partial_tree_decoding(
    model,
    tree_candidates,
    full_past_key_values,
    partial_past_key_values,
    tree_position_ids,
    input_ids,
    retrieve_indices,
):
    prev_full_enabled = full_past_key_values.enabled
    logits, hidden_state, outputs = tree_decoding(
        model,
        tree_candidates,
        full_past_key_values,
        partial_past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
    )
    full_past_key_values.enabled = prev_full_enabled
    return logits, hidden_state, outputs


@torch.inference_mode()
def spec_generate_with_latent_drift(
    model,
    input_ids,
    spec_config,
    max_new_tokens=256,
    max_length=65000,
    temperature=0.0,
    top_p=0.0,
    top_k=0.0,
    is_llama3=True,
):
    if is_llama3:
        stop_token_id = model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    else:
        stop_token_id = None
    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    else:
        logits_processor = None

    padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
    input_ids = input_ids.clone()

    (
        full_past_key_values,
        partial_past_key_values,
        draft_past_key_values,
    ) = model.full_past_key_values, model.partial_past_key_values, model.draft_past_key_values
    full_past_key_values.reset()
    partial_past_key_values.reset()
    draft_past_key_values.reset()
    full_past_key_values.enabled = True

    input_len = input_ids.shape[1]
    reset_tree_mode(model)
    (
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        logits,
        hidden_state,
        sample_token,
    ) = chunked_prefilling(
        input_ids,
        model,
        full_past_key_values,
        draft_past_key_values,
        logits_processor,
    )

    cuda_sync()
    start_time = time.time()

    new_token = 0
    tokens_since_full = 0
    records = []
    verify_mode_counts = {"full_refresh": 0, "partial_shadow": 0}
    max_decode_steps = max_length - model.ea_layer.total_tokens - 10

    for _ in range(max_decode_steps):
        model.base_model.model.tree_mask = tree_mask
        draft_tokens = draft_tokens.to(input_ids.device)

        if (
            input_ids.shape[1] > partial_past_key_values.cache_config.static_kv_size
            and spec_config.enable_partial_kv
        ):
            partial_past_key_values.init_key_values(full_past_key_values)

        use_partial = should_partial_verify(
            partial_past_key_values,
            model.ea_layer.total_tokens,
        )
        if use_partial:
            verify_mode_counts["partial_shadow"] += 1
        else:
            verify_mode_counts["full_refresh"] += 1

        hidden_state_partial = None
        if use_partial:
            _, hidden_state_partial, _ = shadow_partial_tree_decoding(
                model,
                draft_tokens,
                full_past_key_values,
                partial_past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

        logits, hidden_state_new, outputs = full_reference_tree_decoding(
            model,
            draft_tokens,
            full_past_key_values,
            partial_past_key_values,
            tree_position_ids,
            input_ids,
            retrieve_indices,
            refresh_partial_cache=not use_partial,
        )

        draft_tokens = torch.cat((draft_tokens, padding), dim=1)
        candidates = draft_tokens[0, retrieve_indices]

        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits,
            candidates,
            logits_processor,
        )

        n_accepted = int(accept_length.item() if isinstance(accept_length, torch.Tensor) else accept_length) + 1
        if use_partial:
            hidden_full = accepted_hidden_states(
                hidden_state_new,
                retrieve_indices,
                best_candidate,
                accept_length,
            )
            hidden_partial = accepted_hidden_states(
                hidden_state_partial,
                retrieve_indices,
                best_candidate,
                accept_length,
            )
            cosine = F.cosine_similarity(
                hidden_full.float(),
                hidden_partial.float(),
                dim=-1,
            )[0]
            for local_idx, score in enumerate(cosine.tolist()):
                generated_position = int(new_token) + local_idx + 1
                if generated_position > max_new_tokens:
                    continue
                records.append(
                    {
                        "generated_position": generated_position,
                        "distance_since_full": tokens_since_full + local_idx + 1,
                        "cosine": float(score),
                        "accept_length": n_accepted,
                    }
                )
            tokens_since_full += n_accepted
        else:
            tokens_since_full = 0

        (
            input_ids,
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            new_token,
        ) = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            retrieve_indices,
            logits_processor,
            new_token,
            full_past_key_values,
            partial_past_key_values,
            draft_past_key_values,
            model,
            hidden_state_new,
            sample_p,
        )

        if is_llama3 and stop_token_id in input_ids[0, input_len:].tolist():
            break
        if model.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token >= max_new_tokens:
            break
        if input_ids.shape[1] > max_decode_steps:
            break

    cuda_sync()
    total_time = time.time() - start_time
    new_token = new_token.item() if isinstance(new_token, torch.Tensor) else new_token
    return input_ids, {
        "new_token": int(new_token),
        "total_time": total_time,
        "throughput": new_token / total_time if total_time > 0 else 0.0,
        "verify_mode_counts": verify_mode_counts,
        "latent_drift_records": records,
    }


def summarize(records):
    if len(records) == 0:
        return {}
    cos = torch.tensor([record["cosine"] for record in records], dtype=torch.float32)
    return {
        "n_compared_tokens": len(records),
        "mean_cosine": cos.mean().item(),
        "min_cosine": cos.min().item(),
        "p05_cosine": torch.quantile(cos, 0.05).item(),
        "p50_cosine": torch.quantile(cos, 0.50).item(),
        "p95_cosine": torch.quantile(cos, 0.95).item(),
        "mean_drift": (1.0 - cos).mean().item(),
    }


def summarize_by_distance(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["distance_since_full"], []).append(record["cosine"])

    summary = {}
    for distance, values in sorted(grouped.items()):
        cos = torch.tensor(values, dtype=torch.float32)
        summary[str(distance)] = {
            "count": len(values),
            "mean_cosine": cos.mean().item(),
            "min_cosine": cos.min().item(),
        }
    return summary


def run_worker(args, sample_indices, rank=0, world_size=1):
    dataset = load_dataset("parquet", data_files=args.data_path)["train"]
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        device_map = {"": rank}
    else:
        device = torch.device("cpu")
        device_map = "auto"
    model = Speculator.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.draft_model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        total_token=-1,
    )
    model.eval()

    tokenizer = model.tokenizer
    model_name = os.path.basename(args.base_model_path).lower()
    spec_config = build_selector_config(args)
    (
        model.full_past_key_values,
        model.partial_past_key_values,
        model.draft_past_key_values,
    ) = initialize_past_key_values(
        model.base_model,
        model.ea_layer,
        spec_config,
        max_length=args.max_length,
    )
    model.max_length = args.max_length

    sample_metrics = []
    all_records = []
    total_new_tokens = 0
    total_time = 0.0
    verify_mode_counts = {"full_refresh": 0, "partial_shadow": 0}

    desc = "latent drift" if world_size == 1 else f"gpu{rank}"
    for sample_idx in iter_progress(sample_indices, desc, position=rank):
        text = dataset[sample_idx]["text"]
        prompt = build_chat(tokenizer, text, model_name)
        model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
        input_ids = model_inputs["input_ids"][:, : args.context_length]

        output_ids, metrics = spec_generate_with_latent_drift(
            model,
            input_ids,
            spec_config=spec_config,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            temperature=0.0,
            is_llama3="llama3" in model_name,
        )
        sample_records = metrics["latent_drift_records"]
        for record in sample_records:
            record["sample_idx"] = sample_idx
        sample_summary = summarize(sample_records)
        sample_metrics.append(
            {
                "sample_idx": sample_idx,
                "context_length": int(input_ids.shape[-1]),
                "new_token": metrics["new_token"],
                "total_time": metrics["total_time"],
                "throughput": metrics["throughput"],
                "verify_mode_counts": metrics["verify_mode_counts"],
                "summary": sample_summary,
            }
        )
        all_records.extend(sample_records)
        total_new_tokens += metrics["new_token"]
        total_time += metrics["total_time"]
        for key, value in metrics["verify_mode_counts"].items():
            verify_mode_counts[key] = verify_mode_counts.get(key, 0) + value

        del model_inputs, input_ids, output_ids, metrics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "new_token": total_new_tokens,
        "total_time": total_time,
        "throughput": total_new_tokens / total_time if total_time > 0 else 0.0,
        "verify_mode_counts": verify_mode_counts,
        "latent_drift_records": all_records,
        "summary": summarize(all_records),
        "distance_summary": summarize_by_distance(all_records),
        "context_length": args.context_length,
        "partial_length": args.partial_length if args.kv_budget is None else None,
        "kv_selector": args.kv_selector,
        "kv_budget": spec_config.n_sink_blocks * spec_config.block_size
        + spec_config.n_retrieval_blocks * spec_config.block_size
        + spec_config.n_window_blocks * spec_config.block_size,
        "sink_tokens": spec_config.n_sink_blocks * spec_config.block_size,
        "retrieval_tokens": spec_config.n_retrieval_blocks * spec_config.block_size,
        "window_tokens": spec_config.n_window_blocks * spec_config.block_size,
        "partial_spec_tokens": args.partial_spec_tokens,
        "num_samples": len(sample_indices),
        "rank": rank,
        "reference_trajectory": "full",
        "shadow_verifier": "partial",
        "samples": sample_metrics,
    }


def worker_entry(rank, args, sample_indices, world_size, result_queue):
    try:
        metrics = run_worker(args, sample_indices, rank=rank, world_size=world_size)
        result_queue.put({"rank": rank, "metrics": metrics, "error": None})
    except Exception as exc:
        result_queue.put({"rank": rank, "metrics": None, "error": repr(exc)})


def merge_worker_metrics(worker_metrics, args, sample_indices):
    spec_config = build_selector_config(args)
    all_records = []
    sample_metrics = []
    total_new_tokens = 0
    total_time = 0.0
    verify_mode_counts = {"full_refresh": 0, "partial_shadow": 0}

    for metrics in worker_metrics:
        all_records.extend(metrics["latent_drift_records"])
        sample_metrics.extend(metrics["samples"])
        total_new_tokens += metrics["new_token"]
        total_time += metrics["total_time"]
        for key, value in metrics["verify_mode_counts"].items():
            verify_mode_counts[key] = verify_mode_counts.get(key, 0) + value

    all_records.sort(
        key=lambda record: (record.get("sample_idx", -1), record["generated_position"])
    )
    sample_metrics.sort(key=lambda item: item["sample_idx"])

    return {
        "new_token": total_new_tokens,
        "total_time": total_time,
        "throughput": total_new_tokens / total_time if total_time > 0 else 0.0,
        "verify_mode_counts": verify_mode_counts,
        "latent_drift_records": all_records,
        "summary": summarize(all_records),
        "distance_summary": summarize_by_distance(all_records),
        "context_length": args.context_length,
        "partial_length": args.partial_length if args.kv_budget is None else None,
        "kv_selector": args.kv_selector,
        "kv_budget": spec_config.n_sink_blocks * spec_config.block_size
        + spec_config.n_retrieval_blocks * spec_config.block_size
        + spec_config.n_window_blocks * spec_config.block_size,
        "sink_tokens": spec_config.n_sink_blocks * spec_config.block_size,
        "retrieval_tokens": spec_config.n_retrieval_blocks * spec_config.block_size,
        "window_tokens": spec_config.n_window_blocks * spec_config.block_size,
        "partial_spec_tokens": args.partial_spec_tokens,
        "sample_idx": args.sample_idx,
        "num_samples": len(sample_indices),
        "num_gpus": args.num_gpus,
        "reference_trajectory": "full",
        "shadow_verifier": "partial",
        "samples": sample_metrics,
    }


def main():
    args = parse_args()
    dataset = load_dataset("parquet", data_files=args.data_path)["train"]
    if args.num_samples < 1:
        raise ValueError("--num_samples must be at least 1")
    if args.num_gpus < 1:
        raise ValueError("--num_gpus must be at least 1")
    sample_indices = list(
        range(args.sample_idx, min(args.sample_idx + args.num_samples, len(dataset)))
    )
    if len(sample_indices) == 0:
        raise ValueError(
            f"No samples to run: sample_idx={args.sample_idx}, dataset size={len(dataset)}"
        )

    if args.num_gpus == 1:
        metrics = run_worker(args, sample_indices)
        metrics["sample_idx"] = args.sample_idx
        metrics["num_gpus"] = 1
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("--num_gpus > 1 requires CUDA")
        visible_gpus = torch.cuda.device_count()
        if args.num_gpus > visible_gpus:
            raise RuntimeError(
                f"Requested {args.num_gpus} GPUs, but only {visible_gpus} are visible"
            )
        mp.set_start_method("spawn", force=True)
        chunk_size = (len(sample_indices) + args.num_gpus - 1) // args.num_gpus
        worker_indices = [
            sample_indices[rank * chunk_size : (rank + 1) * chunk_size]
            for rank in range(args.num_gpus)
        ]
        for rank, indices in enumerate(worker_indices):
            if len(indices) > 0:
                print(
                    f"rank {rank} -> cuda:{rank}, "
                    f"samples {indices[0]}-{indices[-1]} ({len(indices)} samples)"
                )
        result_queue = mp.Queue()
        processes = []
        for rank, indices in enumerate(worker_indices):
            if len(indices) == 0:
                continue
            process = mp.Process(
                target=worker_entry,
                args=(rank, args, indices, args.num_gpus, result_queue),
            )
            process.start()
            processes.append(process)

        results = []
        for _ in processes:
            result = result_queue.get()
            if result["error"] is not None:
                for process in processes:
                    process.terminate()
                raise RuntimeError(f"Worker {result['rank']} failed: {result['error']}")
            results.append(result)

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker process exited with code {process.exitcode}"
                )

        worker_metrics = [
            result["metrics"] for result in sorted(results, key=lambda item: item["rank"])
        ]
        metrics = merge_worker_metrics(worker_metrics, args, sample_indices)

    print(json.dumps({k: v for k, v in metrics.items() if k != "latent_drift_records"}, indent=2))

    if args.output_path is not None:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved latent drift records to {args.output_path}")


if __name__ == "__main__":
    main()
