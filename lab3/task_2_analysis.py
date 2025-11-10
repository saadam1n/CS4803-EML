import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from collections import defaultdict
import glob

# Hardcoded output directory for plots
OUTPUT_DIR = "./pipeline_plots"
Path(OUTPUT_DIR).mkdir(exist_ok=True)


def get_files_by_prefix(prefix: str) -> List[str]:
    """
    Get all JSON files matching a prefix pattern.
    
    Args:
        prefix: File path prefix (e.g., 'data/varying_microbatch_')
        
    Returns:
        List of matching file paths
    """
    pattern = f"{prefix}*.json"
    files = glob.glob(pattern)
    files.sort()  # Sort for consistent ordering
    
    print(f"\nFound {len(files)} files matching '{pattern}':")
    for f in files:
        print(f"  - {f}")
    
    return files


def save_results(results: List[Dict], output_path: str):
    """
    Save pipeline parallelism results to JSON file.
    
    Args:
        results: List of dictionaries containing seed, throughput, and memory data
        output_path: Path where JSON file will be saved
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def load_json(path: str) -> List[Dict]:
    """Load a single JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def categorize_configs(json_paths: List[str]) -> Dict[str, Dict]:
    """
    Automatically categorize JSON files based on their configuration.
    
    Returns:
        Dictionary mapping each path to its configuration metadata
    """
    configs = {}
    
    for path in json_paths:
        data = load_json(path)
        # Get config from first entry (should be same for all seeds)
        sample = data[0]
        
        config_key = {
            'world_size': sample.get('world_size'),
            'chunks': sample.get('chunks'),
            'microbatch_size': sample.get('microbatch_size'),
            'partition': sample.get('partition'),
            'global_batch_size': sample.get('global_batch_size')
        }
        
        configs[path] = config_key
    
    return configs


def compute_stats(data: List[Dict]) -> Dict:
    """Compute mean and std statistics from results."""
    throughputs = [run['throughput_samples_per_s'] for run in data]
    
    # Memory is per-rank, convert to GB
    mem_per_seed_gb = [[m / (1024**3) for m in run['mem_peak_per_rank_bytes']] 
                        for run in data]
    
    # Compute statistics
    throughput_mean = np.mean(throughputs)
    throughput_std = np.std(throughputs, ddof=1)
    
    # Per-rank memory stats
    mem_per_rank_mean = np.mean(mem_per_seed_gb, axis=0)
    mem_per_rank_std = np.std(mem_per_seed_gb, axis=0, ddof=1)
    
    # Overall memory stats
    all_mem = [m for seed_mem in mem_per_seed_gb for m in seed_mem]
    mem_overall_mean = np.mean(all_mem)
    mem_overall_std = np.std(all_mem, ddof=1)
    
    return {
        'throughput_mean': throughput_mean,
        'throughput_std': throughput_std,
        'mem_per_rank_mean_gb': mem_per_rank_mean,
        'mem_per_rank_std_gb': mem_per_rank_std,
        'mem_overall_mean_gb': mem_overall_mean,
        'mem_overall_std_gb': mem_overall_std
    }


def analyze_deliverable_2(json_paths: List[str]):
    """
    Deliverable 2: Compare 3 vs 4 GPUs
    """
    print("\n" + "="*80)
    print("DELIVERABLE 2: GPU Configuration Comparison")
    print("="*80)
    
    configs = categorize_configs(json_paths)
    
    # Group by world_size (number of GPUs)
    grouped = defaultdict(list)
    for path, config in configs.items():
        world_size = config['world_size']
        grouped[world_size].append(path)
    
    # Analyze each group
    results = []
    for world_size in sorted(grouped.keys()):
        paths = grouped[world_size]
        # Use first path as representative (all should have same config)
        data = load_json(paths[0])
        stats = compute_stats(data)
        
        results.append({
            'Configuration': f'{world_size}_GPUs',
            '# GPUs': world_size,
            'Throughput (samples/s)': f"{stats['throughput_mean']:.3f} ± {stats['throughput_std']:.3f}",
            'Peak Memory per Rank (GB)': f"{stats['mem_overall_mean_gb']:.3f} ± {stats['mem_overall_std_gb']:.3f}"
        })
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    print("\n" + "="*80)
    
    return df


def analyze_deliverable_3(json_paths: List[str]):
    """
    Deliverable 3: Effect of chunks (microbatch size) on throughput and memory
    """
    print("\n" + "="*80)
    print("DELIVERABLE 3: Chunks Analysis")
    print("="*80)
    
    configs = categorize_configs(json_paths)
    
    # Group by world_size, then organize by chunks
    grouped = defaultdict(lambda: defaultdict(list))
    for path, config in configs.items():
        world_size = config['world_size']
        chunks = config['chunks']
        grouped[world_size][chunks].append(path)
    
    # Collect data for all GPU configurations
    all_data = {}
    for world_size in sorted(grouped.keys()):
        chunks_data = grouped[world_size]
        
        chunks_list = []
        throughput_means = []
        throughput_stds = []
        memory_means = []
        memory_stds = []
        
        for chunks in sorted(chunks_data.keys()):
            paths = chunks_data[chunks]
            data = load_json(paths[0])  # Use first path
            stats = compute_stats(data)
            
            chunks_list.append(chunks)
            throughput_means.append(stats['throughput_mean'])
            throughput_stds.append(stats['throughput_std'])
            memory_means.append(stats['mem_overall_mean_gb'])
            memory_stds.append(stats['mem_overall_std_gb'])
        
        all_data[world_size] = {
            'chunks': chunks_list,
            'throughput_means': throughput_means,
            'throughput_stds': throughput_stds,
            'memory_means': memory_means,
            'memory_stds': memory_stds
        }
    
    # Create single figure with two subplots comparing all GPU configs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    # Plot throughput for all GPU configs
    for idx, (world_size, data) in enumerate(sorted(all_data.items())):
        ax1.errorbar(data['chunks'], data['throughput_means'], 
                     yerr=data['throughput_stds'],
                     marker=markers[idx % len(markers)], 
                     capsize=5, linewidth=2, markersize=8,
                     label=f'{world_size} GPUs',
                     color=colors[idx % len(colors)])
    
    ax1.set_xlabel('Number of Chunks', fontsize=12)
    ax1.set_ylabel('Throughput (samples/s)', fontsize=12)
    ax1.set_title('Throughput vs Chunks (All GPU Configs)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot memory for all GPU configs
    for idx, (world_size, data) in enumerate(sorted(all_data.items())):
        ax2.errorbar(data['chunks'], data['memory_means'],
                     yerr=data['memory_stds'],
                     marker=markers[idx % len(markers)],
                     capsize=5, linewidth=2, markersize=8,
                     label=f'{world_size} GPUs',
                     color=colors[idx % len(colors)])
    
    ax2.set_xlabel('Number of Chunks', fontsize=12)
    ax2.set_ylabel('Peak Memory per Rank (GB)', fontsize=12)
    ax2.set_title('Memory vs Chunks (All GPU Configs)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/deliverable3_all_gpus.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()
    
    # Create summary table
    table_data = []
    for world_size in sorted(grouped.keys()):
        chunks_data = grouped[world_size]
        for chunks in sorted(chunks_data.keys()):
            paths = chunks_data[chunks]
            data = load_json(paths[0])
            stats = compute_stats(data)
            
            table_data.append({
                'GPUs': world_size,
                'Chunks': chunks,
                'Throughput (samples/s)': f"{stats['throughput_mean']:.3f} ± {stats['throughput_std']:.3f}",
                'Memory per Rank (GB)': f"{stats['mem_overall_mean_gb']:.3f} ± {stats['mem_overall_std_gb']:.3f}"
            })
    
    df = pd.DataFrame(table_data)
    print("\n" + df.to_string(index=False))
    print("\n" + "="*80)
    
    return df


def analyze_deliverable_4(json_paths: List[str]):
    """
    Deliverable 4: Balanced vs Unbalanced partition comparison
    """
    print("\n" + "="*80)
    print("DELIVERABLE 4: Partition Strategy Comparison")
    print("="*80)
    
    configs = categorize_configs(json_paths)
    
    # Group by world_size and partition type
    grouped = defaultdict(lambda: defaultdict(list))
    for path, config in configs.items():
        world_size = config['world_size']
        partition = config['partition']
        grouped[world_size][partition].append(path)
    
    # Analyze and create visualization
    throughput_results = []
    
    for world_size in sorted(grouped.keys()):
        partition_data = grouped[world_size]
        
        partition_types = []
        throughput_means = []
        throughput_stds = []
        memory_per_rank_data = []
        forward_time_per_rank_data = []
        
        for partition in sorted(partition_data.keys()):
            paths = partition_data[partition]
            data = load_json(paths[0])
            stats = compute_stats(data)
            
            # Compute forward time statistics (convert to seconds for readability)
            forward_times_per_seed = []
            for run in data:
                if 'stage_forward_time_ms_per_rank' in run:
                    # Convert ms to seconds
                    forward_times_per_seed.append([t / 1000.0 for t in run['stage_forward_time_ms_per_rank']])
            
            forward_time_mean = np.mean(forward_times_per_seed, axis=0) if forward_times_per_seed else None
            forward_time_std = np.std(forward_times_per_seed, axis=0, ddof=1) if forward_times_per_seed else None
            
            partition_types.append(partition)
            throughput_means.append(stats['throughput_mean'])
            throughput_stds.append(stats['throughput_std'])
            memory_per_rank_data.append({
                'partition': partition,
                'means': stats['mem_per_rank_mean_gb'],
                'stds': stats['mem_per_rank_std_gb']
            })
            
            if forward_time_mean is not None:
                forward_time_per_rank_data.append({
                    'partition': partition,
                    'means': forward_time_mean,
                    'stds': forward_time_std
                })
            
            throughput_results.append({
                'GPUs': world_size,
                'Partition': partition,
                'Throughput (samples/s)': f"{stats['throughput_mean']:.3f} ± {stats['throughput_std']:.3f}"
            })
        
        # Create comparison plot with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Forward Time per Rank comparison
        if forward_time_per_rank_data:
            for i, time_data in enumerate(forward_time_per_rank_data):
                ranks = np.arange(len(time_data['means']))
                offset = (i - len(forward_time_per_rank_data)/2 + 0.5) * 0.35
                ax1.bar(ranks + offset, time_data['means'], width=0.35, 
                        yerr=time_data['stds'], capsize=3,
                        label=time_data['partition'], alpha=0.7)
            
            ax1.set_xlabel('GPU Rank', fontsize=12)
            ax1.set_ylabel('Forward Time (seconds)', fontsize=12)
            ax1.set_title(f'Forward Time per Rank ({world_size} GPUs)', fontsize=14)
            ax1.set_xticks(np.arange(world_size))
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Memory per Rank comparison
        for i, mem_data in enumerate(memory_per_rank_data):
            ranks = np.arange(len(mem_data['means']))
            offset = (i - len(memory_per_rank_data)/2 + 0.5) * 0.35
            ax2.bar(ranks + offset, mem_data['means'], width=0.35, 
                    yerr=mem_data['stds'], capsize=3,
                    label=mem_data['partition'], alpha=0.7)
        
        ax2.set_xlabel('GPU Rank', fontsize=12)
        ax2.set_ylabel('Peak Memory (GB)', fontsize=12)
        ax2.set_title(f'Memory per Rank ({world_size} GPUs)', fontsize=14)
        ax2.set_xticks(np.arange(world_size))
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Overall Throughput comparison
        x = np.arange(len(partition_types))
        bars = ax3.bar(x, throughput_means, yerr=throughput_stds, capsize=5, 
                alpha=0.7, color=['blue', 'red', 'green', 'orange'][:len(partition_types)])
        ax3.set_xticks(x)
        ax3.set_xticklabels(partition_types, rotation=45, ha='right')
        ax3.set_ylabel('Throughput (samples/s)', fontsize=12)
        ax3.set_title(f'Throughput by Partition ({world_size} GPUs)', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = f"{OUTPUT_DIR}/deliverable4_{world_size}gpus.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
        plt.close()
    
    # Create throughput comparison table
    df = pd.DataFrame(throughput_results)
    print("\n" + "="*80)
    print("THROUGHPUT COMPARISON TABLE")
    print("="*80)
    print("\n" + df.to_string(index=False))
    print("\n" + "="*80)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process pipeline parallelism results')
    parser.add_argument('--mode', choices=['save', 'analyze'], required=True,
                        help='Mode: save results to JSON or analyze multiple JSONs')
    parser.add_argument('--deliverable', type=int, choices=[2, 3, 4],
                        help='Which deliverable to analyze (2, 3, or 4)')
    parser.add_argument('--out', type=str, help='Output JSON path (for save mode)')
    parser.add_argument('--prefix', type=str, 
                        help='File prefix to match (e.g., "data/varying_microbatch_")')
    
    args = parser.parse_args()
    
    if args.mode == 'save':
        if not args.out:
            raise ValueError("--out is required for save mode")
        
        # Example results (replace with your actual results)
        example_results = [
            {
                'world_size': 4,
                'global_batch_size': 64,
                'chunks': 4,
                'microbatch_size': 16,
                'partition': 'even',
                'seed': 1,
                'throughput_samples_per_s': 9.725066062009567,
                'mem_peak_per_rank_bytes': [5580828672, 5479640576, 5479640576, 10519882752]
            },
            {
                'world_size': 4,
                'global_batch_size': 64,
                'chunks': 4,
                'microbatch_size': 16,
                'partition': 'even',
                'seed': 2,
                'throughput_samples_per_s': 8.567090129529957,
                'mem_peak_per_rank_bytes': [5580828672, 5479641088, 5479641088, 10519883264]
            },
            {
                'world_size': 4,
                'global_batch_size': 64,
                'chunks': 4,
                'microbatch_size': 16,
                'partition': 'even',
                'seed': 3,
                'throughput_samples_per_s': 9.407118191760858,
                'mem_peak_per_rank_bytes': [5580828672, 5479641088, 5479641088, 10519883264]
            }
        ]
        
        save_results(example_results, args.out)
    
    elif args.mode == 'analyze':
        if not args.deliverable:
            raise ValueError("--deliverable is required for analyze mode")
        
        # Get JSON paths either from prefix or hardcoded list
        if args.prefix:
            json_paths = get_files_by_prefix(args.prefix)
            if not json_paths:
                raise ValueError(f"No files found matching prefix: {args.prefix}")
        else:
            # HARDCODE YOUR JSON PATHS HERE (if not using --prefix)
            # Add all your JSON files - the script will automatically categorize them
            json_paths = [
                'results_3gpu.json',
                'results_4gpu.json',
                'results_4gpu_chunks2.json',
                'results_4gpu_chunks4.json',
                'results_4gpu_chunks8.json',
                'results_4gpu_even.json',
                'results_4gpu_unbalanced.json',
                # Add more paths as needed
            ]
        
        if args.deliverable == 2:
            df = analyze_deliverable_2(json_paths)
        elif args.deliverable == 3:
            df = analyze_deliverable_3(json_paths)
        elif args.deliverable == 4:
            df = analyze_deliverable_4(json_paths)
        
        print(f"\nAll plots saved to: {OUTPUT_DIR}/")