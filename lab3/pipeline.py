# pipeline.py
# Distributed Pipeline Parallelism — Student Template
#
# Quick sanity check (one step):
#   torchrun --standalone --nproc-per-node=4 pipeline.py  --world_size 4 --global_batch_size 8 --chunks 4 --quick_demo
   
#
# For the assignment:
#   - Implement ALL TODOs below (measurement, seeds aggregation, custom partitioning, optional stage timing).
#   - Then run your experiments for Steps 1–4 and produce tables/plots.
#
# Notes:
#   - Use mean ± std over 3 seeds across Steps 1–4 (same protocol as the pipeline task).
#   - Warm up 20 steps, then measure >= 100 steps (configurable via CLI).

import argparse
import os
import time
import json

import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint
from transformers import BertModel, BertConfig

from hf_utils import generate_inputs_for_model, get_number_of_params


# ---------------------------
# Split spec helpers
# ---------------------------

def make_split_spec_even(model: BertModel, world_size: int):
    """
    Build split points at 'encoder.layer.{k}' to roughly balance blocks per rank.

    """
    n_layers = model.config.num_hidden_layers
    per = (n_layers + world_size - 1) // world_size  # ceil
    if per == 0:
        raise ValueError(f"world_size ({world_size}) cannot exceed num_hidden_layers ({n_layers})")
    return {
        f"encoder.layer.{i * per}": SplitPoint.BEGINNING
        for i in range(1, world_size)
    }


def make_split_spec_custom(model: BertModel, pattern: str, world_size: int):
    """
    TODO (Step 4): parse a custom pattern like "6-2-2-2" and produce split points.
      - Validate: number of parts == world_size
      - Validate: sum(parts) == num_hidden_layers
      - Compute cumulative sums c1, c2, c3,... and place SplitPoint.BEGINNING
        at 'encoder.layer.{c1}', 'encoder.layer.{c2}', ...
    Return a dict[str, SplitPoint].
    """
    # >>> YOUR CODE HERE <<<

    cur_split_pos = 0
    return {
        f"encoder.layer.{(cur_split_pos := cur_split_pos + size)}": SplitPoint.BEGINNING
        for size in pattern[:-1]
    }

    #raise NotImplementedError("Implement custom unbalanced partitioning (e.g., 3-1-2-2).")


# ---------------------------
# Optional: simple per-stage timing (hint)
# ---------------------------

def wrap_stage_forward_for_timing(module: torch.nn.Module, device: torch.device):
    """
    OPTIONAL (Step 4): wrap stage forward to collect avg forward time per step.
    Use CUDA events; store accumulator in module._acc_ms.
    """
    # >>> OPTIONAL: YOUR CODE HERE <<<
    # Hints:
    #   start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    #   with torch.cuda.device(device): start.record(); out = orig(*args, **kw); end.record()
    #   torch.cuda.synchronize(device); module._acc_ms += start.elapsed_time(end)
    # Initialize the accumulator attribute if it doesn't exist
    if not hasattr(module, '_acc_ms'):
        module._acc_ms = 0.0
    
    # Store the original forward method
    orig_forward = module.forward
    
    def timed_forward(*args, **kwargs):
        # Create CUDA events for timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # Record timing around the forward pass
        with torch.cuda.device(device):
            start.record()
            output = orig_forward(*args, **kwargs)
            end.record()
        
        # Synchronize and accumulate elapsed time
        torch.cuda.synchronize(device)
        elapsed_ms = start.elapsed_time(end)
        module._acc_ms += elapsed_ms
        
        return output
    
    # Replace the forward method with the timed version
    module.forward = timed_forward


# ---------------------------
# Warmup + measurement loop
# ---------------------------

def warmup_and_measure(schedule: ScheduleGPipe,
                       model: BertModel,
                       device: torch.device,
                       global_batch_size: int,
                       chunks: int,
                       steps_warmup: int,
                       steps_measure: int):
    """
    TODO (Step 1, Step 2, Step 3): implement full measurement.

    Requirements:
      - Assert global_batch_size % chunks == 0 (integer microbatch size).
      - Warm up for `steps_warmup` steps (do not time).
      - Measure for `steps_measure` steps:
          * Generate inputs each step on rank 0's device
          * Rank 0: schedule.step(**inputs); other ranks: schedule.step()
          * Use dist.barrier() + torch.cuda.synchronize() before/after timing
      - Throughput (samples/s) = (global_batch_size * steps_measure) / elapsed_seconds
      - Peak memory per rank via torch.cuda.max_memory_allocated(rank)
      - Return: (throughput: float, peaks_bytes: list[int])

    Hints:
      - Reset memory stats before timing on each rank: torch.cuda.reset_peak_memory_stats(rank)
      - Use time.time() wall clock for elapsed_seconds
    """
    # >>> YOUR CODE HERE <<<
    rank = dist.get_rank()

    torch.cuda.reset_peak_memory_stats(rank)

    if rank == 0:
        full_inputs = generate_inputs_for_model(BertModel, model, "BertModel", global_batch_size, device)    

    num_warmup_iters = steps_warmup
    num_timing_iters = steps_measure

    for i in range(num_warmup_iters + num_timing_iters):
        if i < num_warmup_iters:
            if rank == 0:
                print(f"Completing warmup iteration {i+1}/{num_warmup_iters}")
        elif i == num_warmup_iters:
            if rank == 0:
                print("Warmup completed, starting data collection...")

            tic = time.time()

        if rank == 0:
            schedule.step(**full_inputs)
        else:
            _ = schedule.step()
        dist.barrier()
    
    torch.cuda.synchronize(device)
    toc = time.time()

    if rank == 0:
        print("Data collection completed")

    throughput = num_timing_iters / (toc - tic)
    peak_bytes = torch.cuda.max_memory_allocated(rank)



    return throughput, peak_bytes

    #raise NotImplementedError("Implement warmup+measure timing and peak memory collection.")


# ---------------------------
# One run for a given seed
# ---------------------------

def run_one_seed(args, pipe_ir, model, seed):
    """
    TODO: If you want per-stage forward timing (Step 4), wrap your stage module first.

    Return a dict with at least:
      - 'seed': int
      - 'throughput_samples_per_s': float
      - 'mem_peak_per_rank_bytes': List[int]
      - (optional) 'stage_forward_time_ms_per_rank': List[float or None]
    """
    rank = dist.get_rank()

    # Build runtime schedule
    stage = pipe_ir.build_stage(args.rank, device=args.device)
    schedule = ScheduleGPipe(stage, args.chunks)

    # OPTIONAL: per-stage forward timing
    stage_module = pipe_ir.get_stage_module(rank)
    wrap_stage_forward_for_timing(stage_module, args.device)

    # Warmup + measure
    thr, peaks = warmup_and_measure(
        schedule=schedule,
        model=model,
        device=args.device,
        global_batch_size=args.global_batch_size,
        chunks=args.chunks,
        steps_warmup=args.steps_warmup,
        steps_measure=args.steps_measure,
    )

    # OPTIONAL: stage timing average per step if you wrapped forward
    stage_ms = getattr(stage_module, "_acc_ms", None)
    if stage_ms is not None:
        stage_ms = float(stage_ms) / args.steps_measure

    # Gather per-rank peaks to rank 0
    # (Hint: use dist.all_gather_object to collect Python ints from all ranks)
    # >>> YOUR CODE HERE <<<  (replace the fake single-rank list below)
    
    peaks_all = [None] * dist.get_world_size() 
    torch.cuda.set_device(rank) 
    dist.all_gather_object(peaks_all, peaks)

    stage_all = [None] * dist.get_world_size() 
    dist.all_gather_object(stage_all, stage_ms)

    out = {
        "world_size": dist.get_world_size(),
        "global_batch_size": args.global_batch_size,
        "chunks": args.chunks,
        "microbatch_size": args.global_batch_size // args.chunks,
        "partition": args.partition,
        "seed": seed,
        "throughput_samples_per_s": float(thr),
        "mem_peak_per_rank_bytes": peaks_all,
        "stage_forward_time_ms_per_rank": stage_all
    }
    return out


# ---------------------------
# Main
# ---------------------------

def run(args):
    rank = args.rank
    world_size = args.world_size

    config = BertConfig()
    config.return_dict = False

    # Create model
    model = BertModel(config)
    model.to(args.device)
    model.eval()

    if rank == 0:
        print(model)
        print(model.config)
        print(f"Total params ≈ {get_number_of_params(model) // 10 ** 6}M")

    # Example microbatch
    assert args.global_batch_size % args.chunks == 0, "global_batch_size must be divisible by chunks"
    example_mb = generate_inputs_for_model(BertModel, model, "BertModel",
                                           args.global_batch_size // args.chunks, args.device)

    # Split points
    if args.partition == "even":
        split_spec = make_split_spec_even(model, world_size)
    else:
        parsed_part = [int(x) for x in args.partition.split(',')]

        split_spec = make_split_spec_custom(model, parsed_part, world_size)

    if rank == 0:
        print("Split points:", list(split_spec.keys()))

    # Build pipeline IR
    pipe_ir = pipeline(
        model,
        mb_args=(),
        mb_kwargs=example_mb,
        split_spec=split_spec,
    )
    assert pipe_ir.num_stages == world_size, f"nstages={pipe_ir.num_stages} != world_size={world_size}"

    # Quick test (one step)
    if args.quick_demo:
        stage = pipe_ir.build_stage(args.rank, device=args.device)
        schedule = ScheduleGPipe(stage, args.chunks)
        full_inputs = generate_inputs_for_model(BertModel, model, "BertModel",
                                                args.global_batch_size, args.device)
        if rank == 0:
            schedule.step(**full_inputs)
        else:
            _ = schedule.step()
        dist.barrier()
        print(f"[Rank {rank}] quick_demo complete")
        return

    # ===== Multi-seed measurements (Step 1 / Step 2 / Step 3 / Step 4) =====
    # TODO: loop over args.seeds, set torch.manual_seed/torch.cuda.manual_seed_all,
    #       call run_one_seed each time, aggregate mean ± std on rank 0,
    #       and optionally save JSON to args.out for plotting.
    #
    # Hints:
    #   - Parse seeds from comma-separated string (e.g., "1,2,3").
    #   - On rank 0, compute mean/std for throughput; for memory, compute mean/std per rank.
    #   - If args.out is set, write a dict containing spec + per_seed + agg results.
    #
    # >>> YOUR CODE HERE <<<

    parsed_seeds = [int(x) for x in args.seeds.split(',')]

    if rank == 0:
        print(f"Utilizing seeds {parsed_seeds}")

    perf_results = []
    for seed in parsed_seeds:
        if rank == 0:
            print(f"Using seed {seed}")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        cur_results = run_one_seed(args, pipe_ir, model, seed)

        perf_results.append(cur_results)

    print(f"[Rank {rank}] Results: {perf_results}")

    # TODO: save to json
    if rank == 0 and args.out is not None:
        with open(args.out, 'w') as f:
            json.dump(perf_results, f, indent=2)
            
        print(f"[Rank {rank}] Saved results to {args.out}")
    elif rank == 0:
        print(f"[Rank {rank}] No output directory! Please note this data will be unuseable in analysis.")


    #raise NotImplementedError("Implement multi-seed loop, aggregation, and optional JSON saving.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    p.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    p.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    p.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))

    p.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    p.add_argument('--global_batch_size', type=int, default=64)
    p.add_argument('--chunks', type=int, default=4)
    p.add_argument('--steps_warmup', type=int, default=20)
    p.add_argument('--steps_measure', type=int, default=100)
    p.add_argument('--partition', type=str, default="even",
                   help='either "even" or a custom pattern like "3-1-2-2" (Step 4)')
    p.add_argument('--seeds', type=str, default="1,2,3",
                   help='comma-separated list, e.g., "1,2,3"')
    p.add_argument('--out', type=str, default="",
                   help="optional JSON path to save results (for tables/plots)")
    p.add_argument('--quick_demo', action='store_true', help="run a single untimed step to sanity-check PiPPy")

    args = p.parse_args()

    if args.cuda:
        dev_id = args.rank % max(1, torch.cuda.device_count())
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")

    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(backend=backend, rank=args.rank, world_size=args.world_size)

    try:
        run(args)
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()