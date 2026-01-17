"""
Ablation Study Framework for NER Models.

This script runs systematic experiments to measure the contribution
of each component to the overall model performance.

Experiments:
1. BiLSTM only (baseline - no CRF)
2. BiLSTM + CRF
3. BiLSTM + CRF + Character CNN
4. BiLSTM + CRF + Self-Attention
5. BiLSTM + CRF + Char CNN + Attention
6. Full model + Pre-trained embeddings (GloVe)

Each experiment is run multiple times (default 3) for statistical significance.
Results are saved as JSON and visualized as comparison charts.

Usage:
    python scripts/ablation_study.py --config config/config.yaml --output results/ablation

Built from scratch for NLP Course Project.

Author: NLP Course Project
"""

import os
import sys
import argparse
import yaml
import json
import copy
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train import train, set_seed


def create_experiment_configs(base_config: Dict) -> Dict[str, Dict]:
    """
    Create configurations for each ablation experiment.

    Args:
        base_config: Base configuration dictionary

    Returns:
        Dictionary mapping experiment name to config
    """
    experiments = {}

    # Experiment 1: BiLSTM only (no CRF)
    config1 = copy.deepcopy(base_config)
    config1['model']['use_char_features'] = False
    config1['model']['use_attention'] = False
    config1['model']['use_pretrained_embeddings'] = False
    experiments['1_BiLSTM_only'] = {
        'config': config1,
        'model_type': 'baseline_bilstm',
        'description': 'BiLSTM without CRF (baseline)'
    }

    # Experiment 2: BiLSTM + CRF
    config2 = copy.deepcopy(base_config)
    config2['model']['use_char_features'] = False
    config2['model']['use_attention'] = False
    config2['model']['use_pretrained_embeddings'] = False
    experiments['2_BiLSTM_CRF'] = {
        'config': config2,
        'model_type': 'bilstm_crf',
        'description': 'BiLSTM with CRF layer'
    }

    # Experiment 3: BiLSTM + CRF + Character CNN
    config3 = copy.deepcopy(base_config)
    config3['model']['use_char_features'] = True
    config3['model']['use_attention'] = False
    config3['model']['use_pretrained_embeddings'] = False
    experiments['3_BiLSTM_CRF_CharCNN'] = {
        'config': config3,
        'model_type': 'bilstm_crf',
        'description': 'BiLSTM + CRF + Character CNN'
    }

    # Experiment 4: BiLSTM + CRF + Self-Attention
    config4 = copy.deepcopy(base_config)
    config4['model']['use_char_features'] = False
    config4['model']['use_attention'] = True
    config4['model']['use_pretrained_embeddings'] = False
    experiments['4_BiLSTM_CRF_Attention'] = {
        'config': config4,
        'model_type': 'bilstm_crf',
        'description': 'BiLSTM + CRF + Self-Attention'
    }

    # Experiment 5: BiLSTM + CRF + CharCNN + Attention
    config5 = copy.deepcopy(base_config)
    config5['model']['use_char_features'] = True
    config5['model']['use_attention'] = True
    config5['model']['use_pretrained_embeddings'] = False
    experiments['5_Full_no_pretrained'] = {
        'config': config5,
        'model_type': 'bilstm_crf',
        'description': 'Full model without pre-trained embeddings'
    }

    # Experiment 6: Full model + Pre-trained embeddings
    config6 = copy.deepcopy(base_config)
    config6['model']['use_char_features'] = True
    config6['model']['use_attention'] = True
    config6['model']['use_pretrained_embeddings'] = True
    experiments['6_Full_with_GloVe'] = {
        'config': config6,
        'model_type': 'bilstm_crf',
        'description': 'Full model with GloVe embeddings'
    }

    return experiments


def run_single_experiment(
    config: Dict,
    model_type: str,
    output_dir: str,
    seed: int
) -> Tuple[float, Dict]:
    """
    Run a single experiment.

    Args:
        config: Configuration dictionary
        model_type: Model type to train
        output_dir: Directory to save outputs
        seed: Random seed

    Returns:
        Tuple of (best_f1, history)
    """
    # Set seed
    config['random_seed'] = seed

    # Update output paths
    config['artifacts']['save_dir'] = output_dir
    config['artifacts']['model_checkpoint'] = os.path.join(output_dir, 'model.pt')
    config['artifacts']['vocab_file'] = os.path.join(output_dir, 'vocab.pkl')

    # Save config for this run
    config_path = os.path.join(output_dir, 'config.yaml')
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run training
    try:
        best_f1, history = train(config_path, model_type)
        return best_f1, history
    except Exception as e:
        print(f"Error in experiment: {e}")
        return 0.0, {}


def run_ablation_study(
    base_config_path: str,
    output_dir: str,
    num_runs: int = 3,
    seeds: Optional[List[int]] = None
) -> Dict:
    """
    Run complete ablation study.

    Args:
        base_config_path: Path to base configuration file
        output_dir: Directory to save all outputs
        num_runs: Number of runs per experiment
        seeds: List of random seeds (default: [42, 123, 456])

    Returns:
        Dictionary with all results
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Default seeds
    if seeds is None:
        seeds = [42, 123, 456][:num_runs]

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    study_dir = os.path.join(output_dir, f'ablation_study_{timestamp}')
    os.makedirs(study_dir, exist_ok=True)

    # Create experiment configs
    experiments = create_experiment_configs(base_config)

    # Results storage
    all_results = {
        'study_info': {
            'timestamp': timestamp,
            'num_runs': num_runs,
            'seeds': seeds,
            'base_config': base_config_path
        },
        'experiments': {}
    }

    print("\n" + "=" * 80)
    print("ABLATION STUDY")
    print("=" * 80)
    print(f"Output directory: {study_dir}")
    print(f"Number of runs per experiment: {num_runs}")
    print(f"Seeds: {seeds}")
    print(f"Experiments: {len(experiments)}")
    print("=" * 80 + "\n")

    # Run each experiment
    for exp_name, exp_info in experiments.items():
        print(f"\n{'=' * 60}")
        print(f"Experiment: {exp_name}")
        print(f"Description: {exp_info['description']}")
        print(f"{'=' * 60}\n")

        exp_dir = os.path.join(study_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        f1_scores = []
        histories = []

        for run_idx, seed in enumerate(seeds):
            print(f"\n--- Run {run_idx + 1}/{num_runs} (seed={seed}) ---\n")

            run_dir = os.path.join(exp_dir, f'run_{run_idx + 1}_seed_{seed}')

            start_time = time.time()
            best_f1, history = run_single_experiment(
                exp_info['config'],
                exp_info['model_type'],
                run_dir,
                seed
            )
            elapsed = time.time() - start_time

            f1_scores.append(best_f1)
            histories.append(history)

            print(f"\nRun {run_idx + 1} complete: F1 = {best_f1:.4f}, Time = {elapsed:.1f}s")

        # Compute statistics
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        max_f1 = np.max(f1_scores)
        min_f1 = np.min(f1_scores)

        all_results['experiments'][exp_name] = {
            'description': exp_info['description'],
            'model_type': exp_info['model_type'],
            'f1_scores': f1_scores,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'max_f1': max_f1,
            'min_f1': min_f1,
            'precision': np.mean([h.get('dev_precision', [0])[-1] for h in histories if h]),
            'recall': np.mean([h.get('dev_recall', [0])[-1] for h in histories if h])
        }

        print(f"\n{exp_name} Summary:")
        print(f"  Mean F1: {mean_f1:.4f} +/- {std_f1:.4f}")
        print(f"  Range: [{min_f1:.4f}, {max_f1:.4f}]")

    # Save results
    results_path = os.path.join(study_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Generate summary report
    generate_summary_report(all_results, study_dir)

    # Generate visualization
    try:
        from src.utils.visualization import plot_ablation_results
        plot_data = {
            name: {
                'f1': data['mean_f1'],
                'precision': data['precision'],
                'recall': data['recall']
            }
            for name, data in all_results['experiments'].items()
        }
        plot_ablation_results(
            plot_data,
            os.path.join(study_dir, 'ablation_comparison.png'),
            'Ablation Study Results'
        )
    except Exception as e:
        print(f"Could not generate visualization: {e}")

    return all_results


def generate_summary_report(results: Dict, output_dir: str) -> None:
    """
    Generate a text summary report.

    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    report_path = os.path.join(output_dir, 'summary_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ABLATION STUDY SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Date: {results['study_info']['timestamp']}\n")
        f.write(f"Number of runs per experiment: {results['study_info']['num_runs']}\n")
        f.write(f"Seeds used: {results['study_info']['seeds']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("RESULTS TABLE\n")
        f.write("-" * 80 + "\n\n")

        # Header
        f.write(f"{'Experiment':<30} {'Mean F1':<12} {'Std':<10} {'Max F1':<10} {'Min F1':<10}\n")
        f.write("-" * 80 + "\n")

        # Sort by mean F1
        sorted_exps = sorted(
            results['experiments'].items(),
            key=lambda x: x[1]['mean_f1'],
            reverse=True
        )

        for exp_name, data in sorted_exps:
            f.write(f"{exp_name:<30} {data['mean_f1']:.4f}       "
                   f"{data['std_f1']:.4f}     {data['max_f1']:.4f}     {data['min_f1']:.4f}\n")

        f.write("-" * 80 + "\n\n")

        # Component contribution analysis
        f.write("COMPONENT CONTRIBUTION ANALYSIS\n")
        f.write("-" * 80 + "\n\n")

        baseline = results['experiments'].get('1_BiLSTM_only', {}).get('mean_f1', 0)

        for exp_name, data in sorted_exps:
            improvement = data['mean_f1'] - baseline
            f.write(f"{exp_name}:\n")
            f.write(f"  Description: {data['description']}\n")
            f.write(f"  F1: {data['mean_f1']:.4f} (+{improvement:.4f} from baseline)\n")
            f.write(f"  Precision: {data['precision']:.4f}\n")
            f.write(f"  Recall: {data['recall']:.4f}\n\n")

        # Key findings
        f.write("\nKEY FINDINGS\n")
        f.write("-" * 80 + "\n\n")

        best_exp = sorted_exps[0]
        f.write(f"1. Best performing model: {best_exp[0]}\n")
        f.write(f"   F1 Score: {best_exp[1]['mean_f1']:.4f}\n\n")

        # CRF contribution
        bilstm_only = results['experiments'].get('1_BiLSTM_only', {}).get('mean_f1', 0)
        bilstm_crf = results['experiments'].get('2_BiLSTM_CRF', {}).get('mean_f1', 0)
        crf_contribution = bilstm_crf - bilstm_only
        f.write(f"2. CRF contribution: +{crf_contribution:.4f} F1\n\n")

        # Character CNN contribution
        with_char = results['experiments'].get('3_BiLSTM_CRF_CharCNN', {}).get('mean_f1', 0)
        char_contribution = with_char - bilstm_crf
        f.write(f"3. Character CNN contribution: +{char_contribution:.4f} F1\n\n")

        # Attention contribution
        with_attn = results['experiments'].get('4_BiLSTM_CRF_Attention', {}).get('mean_f1', 0)
        attn_contribution = with_attn - bilstm_crf
        f.write(f"4. Self-Attention contribution: +{attn_contribution:.4f} F1\n\n")

        # Pre-trained embeddings contribution
        full_no_pretrained = results['experiments'].get('5_Full_no_pretrained', {}).get('mean_f1', 0)
        full_with_glove = results['experiments'].get('6_Full_with_GloVe', {}).get('mean_f1', 0)
        pretrained_contribution = full_with_glove - full_no_pretrained
        f.write(f"5. Pre-trained embeddings (GloVe) contribution: +{pretrained_contribution:.4f} F1\n\n")

        f.write("=" * 80 + "\n")

    print(f"Summary report saved to {report_path}")


def quick_ablation(base_config_path: str, output_dir: str) -> Dict:
    """
    Run a quick ablation study with fewer epochs for testing.

    Args:
        base_config_path: Path to base configuration
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    # Load and modify config for quick run
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Reduce epochs for quick testing
    base_config['training']['num_epochs'] = 3
    base_config['training']['early_stopping'] = False

    # Save modified config
    quick_config_path = os.path.join(output_dir, 'quick_config.yaml')
    os.makedirs(output_dir, exist_ok=True)

    with open(quick_config_path, 'w') as f:
        yaml.dump(base_config, f)

    # Run with single seed
    return run_ablation_study(quick_config_path, output_dir, num_runs=1, seeds=[42])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run ablation study for NER model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to base config file')
    parser.add_argument('--output', type=str, default='results/ablation',
                        help='Output directory for results')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of runs per experiment')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds for runs')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick ablation (fewer epochs, single run)')

    args = parser.parse_args()

    if args.quick:
        results = quick_ablation(args.config, args.output)
    else:
        results = run_ablation_study(
            args.config,
            args.output,
            num_runs=args.num_runs,
            seeds=args.seeds
        )

    print("\nAblation study complete!")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
