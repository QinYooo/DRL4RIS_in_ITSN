"""
Parse training logs from train_rl_with_ae.py
Extracts key metrics and generates summary reports
"""

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from datetime import datetime


def parse_console_log(log_path):
    """
    Parse console output log file

    Args:
        log_path: Path to log file

    Returns:
        dict: Parsed data including config, baseline, trajectory_evals
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = {
        'config': {},
        'baseline': None,
        'trajectory_evals': [],
        'final_eval': None
    }

    # State machine for parsing
    section = None
    current_trajectory = None

    for i, line in enumerate(lines):
        line = line.strip()

        # Parse configuration section
        if "Configuration:" in line:
            section = 'config'
            continue

        if section == 'config':
            if line.startswith('  AE checkpoint:'):
                result['config']['ae_checkpoint'] = line.split(':')[1].strip()
            elif line.startswith('  Total timesteps:'):
                result['config']['total_timesteps'] = int(line.split(':')[1].strip().replace(',', ''))
            elif line.startswith('  Parallel envs:'):
                result['config']['n_envs'] = int(line.split(':')[1].strip())
            elif line.startswith('  Learning rate:'):
                result['config']['learning_rate'] = float(line.split(':')[1].strip())
            elif line.startswith('  Device:'):
                result['config']['device'] = line.split(':')[1].strip()
            elif line.startswith('  Seed:'):
                result['config']['seed'] = int(line.split(':')[1].strip())
            elif "===" in line:
                section = None

        # Parse baseline evaluation
        if "Baseline Evaluation (Zero Action, seed=" in line:
            section = 'baseline'
            current_baseline = {'steps': []}
            continue

        if section == 'baseline':
            if line.startswith('Step ') and '|' in line:
                continue  # Skip header
            if line.startswith('-'):
                continue
            # Parse data line: "   0 |  1.2345 |  5.6789 |  6.9134 |      -5.23 |   -1.234 |       Y"
            if re.match(r'^\s*\d+\s+\|', line):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 6:
                    try:
                        step_data = {
                            'step': int(parts[0]),
                            'P_BS': float(parts[1]),
                            'P_sat': float(parts[2]),
                            'P_total': float(parts[3]),
                            'SINR_min': float(parts[4]),
                            'reward': float(parts[5]),
                            'success': parts[6].strip() == 'Y'
                        }
                        current_baseline['steps'].append(step_data)
                    except (ValueError, IndexError):
                        pass
            if "Baseline Summary:" in line:
                section = 'baseline_summary'
            if "===" in line and len(current_baseline['steps']) > 0:
                # Compute summary
                steps_data = current_baseline['steps']
                if steps_data:
                    current_baseline['summary'] = {
                        'total_steps': len(steps_data),
                        'avg_P_BS': np.mean([s['P_BS'] for s in steps_data]),
                        'avg_P_sat': np.mean([s['P_sat'] for s in steps_data]),
                        'avg_P_total': np.mean([s['P_total'] for s in steps_data]),
                        'avg_SINR_min': np.mean([s['SINR_min'] for s in steps_data]),
                        'success_rate': np.mean([1 if s['success'] else 0 for s in steps_data])
                    }
                result['baseline'] = current_baseline
                section = None
                current_baseline = None

        if section == 'baseline_summary':
            if line.startswith('  Total steps:'):
                pass  # Already computed
            elif "===" in line:
                section = None

        # Parse trajectory evaluation
        if "Evaluating on trajectory (seed=" in line:
            section = 'trajectory'
            current_trajectory = {'steps': []}
            # Extract seed from line
            match = re.search(r'seed=(\d+)', line)
            if match:
                current_trajectory['seed'] = int(match.group(1))
            # Check if it's callback
            if i > 0 and "Callback]" in lines[i-1]:
                current_trajectory['type'] = 'callback'
            else:
                current_trajectory['type'] = 'final'
            continue

        if section == 'trajectory':
            if line.startswith('Step ') and '|' in line:
                continue  # Skip header
            if line.startswith('-'):
                continue
            if re.match(r'^\s*\d+\s+\|', line):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 6:
                    try:
                        step_data = {
                            'step': int(parts[0]),
                            'P_BS': float(parts[1]),
                            'P_sat': float(parts[2]),
                            'P_total': float(parts[3]),
                            'SINR_min': float(parts[4]),
                            'success': parts[5].strip() == '✓'
                        }
                        current_trajectory['steps'].append(step_data)
                    except (ValueError, IndexError):
                        pass
            if "Summary:" in line:
                section = 'trajectory_summary'
            if "===" in line and len(current_trajectory['steps']) > 0:
                # Compute summary
                steps_data = current_trajectory['steps']
                if steps_data:
                    current_trajectory['summary'] = {
                        'total_steps': len(steps_data),
                        'avg_P_BS': np.mean([s['P_BS'] for s in steps_data]),
                        'avg_P_sat': np.mean([s['P_sat'] for s in steps_data]),
                        'avg_P_total': np.mean([s['P_total'] for s in steps_data]),
                        'avg_SINR_min': np.mean([s['SINR_min'] for s in steps_data]),
                        'success_rate': np.mean([1 if s['success'] else 0 for s in steps_data])
                    }
                if current_trajectory['type'] == 'final':
                    result['final_eval'] = current_trajectory
                else:
                    result['trajectory_evals'].append(current_trajectory)
                section = None
                current_trajectory = None

        if section == 'trajectory_summary':
            if "===" in line:
                section = None

    return result


def parse_json_results(results_dir):
    """
    Parse saved JSON result files

    Args:
        results_dir: Path to logs directory containing JSON files

    Returns:
        dict: Parsed data from JSON files
    """
    results_dir = Path(results_dir)
    result = {
        'baseline': None,
        'trajectory_evals': [],
        'final_eval': None
    }

    # Parse baseline
    baseline_file = results_dir / 'baseline_eval.json'
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            result['baseline'] = json.load(f)

    # Parse trajectory evals
    trajectory_dir = results_dir / 'trajectory_evals'
    if trajectory_dir.exists():
        for file in sorted(trajectory_dir.glob('trajectory_eval_*.json')):
            with open(file, 'r') as f:
                data = json.load(f)
                data['source_file'] = file.name
                result['trajectory_evals'].append(data)

    # Parse final eval
    final_file = results_dir / 'final_trajectory_eval.json'
    if final_file.exists():
        with open(final_file, 'r') as f:
            result['final_eval'] = json.load(f)

    return result


def print_summary(parsed_data):
    """Print formatted summary of parsed data"""

    print("=" * 70)
    print("TRAINING LOG SUMMARY")
    print("=" * 70)

    # Configuration
    if parsed_data.get('config'):
        print("\n[Configuration]")
        for key, value in parsed_data['config'].items():
            print(f"  {key}: {value}")

    # Baseline
    if parsed_data.get('baseline'):
        print("\n[Baseline Evaluation (Zero Action)]")
        if 'summary' in parsed_data['baseline']:
            summary = parsed_data['baseline']['summary']
            print(f"  Total steps: {summary['total_steps']}")
            print(f"  Avg P_BS: {summary['avg_P_BS']:.4f} W")
            print(f"  Avg P_sat: {summary['avg_P_sat']:.4f} W")
            print(f"  Avg P_total: {summary['avg_P_total']:.4f} W")
            print(f"  Avg SINR_min: {summary['avg_SINR_min']:.2f} dB")
            print(f"  Success rate: {summary['success_rate']*100:.1f}%")

    # Trajectory evaluations during training
    if parsed_data.get('trajectory_evals'):
        print(f"\n[Trajectory Evaluations (N={len(parsed_data['trajectory_evals'])})]")
        for i, traj in enumerate(parsed_data['trajectory_evals']):
            if 'summary' in traj:
                summary = traj['summary']
                print(f"\n  Evaluation #{i+1} (seed={traj.get('seed', 'N/A')}):")
                print(f"    Avg P_total: {summary['avg_P_total']:.4f} W")
                print(f"    Avg SINR_min: {summary['avg_SINR_min']:.2f} dB")
                print(f"    Success rate: {summary['success_rate']*100:.1f}%")

    # Final evaluation
    if parsed_data.get('final_eval'):
        print("\n[Final Evaluation]")
        if 'summary' in parsed_data['final_eval']:
            summary = parsed_data['final_eval']['summary']
            print(f"  Total steps: {summary['total_steps']}")
            print(f"  Avg P_BS: {summary['avg_P_BS']:.4f} W")
            print(f"  Avg P_sat: {summary['avg_P_sat']:.4f} W")
            print(f"  Avg P_total: {summary['avg_P_total']:.4f} W")
            print(f"  Avg SINR_min: {summary['avg_SINR_min']:.2f} dB")
            print(f"  Success rate: {summary['success_rate']*100:.1f}%")

    # Comparison with baseline
    if parsed_data.get('baseline') and parsed_data.get('final_eval'):
        baseline_summary = parsed_data['baseline'].get('summary', {})
        final_summary = parsed_data['final_eval'].get('summary', {})

        if baseline_summary and final_summary:
            print("\n[Improvement over Baseline]")
            power_improvement = (baseline_summary['avg_P_total'] - final_summary['avg_P_total']) / baseline_summary['avg_P_total'] * 100
            sinr_improvement = final_summary['avg_SINR_min'] - baseline_summary['avg_SINR_min']
            success_improvement = (final_summary['success_rate'] - baseline_summary['success_rate']) * 100

            print(f"  Power reduction: {power_improvement:+.1f}%")
            print(f"  SINR improvement: {sinr_improvement:+.2f} dB")
            print(f"  Success rate change: {success_improvement:+.1f}%")

    print("\n" + "=" * 70)


def extract_training_curves(parsed_data):
    """
    Extract training curves from multiple trajectory evaluations

    Returns:
        dict: Training curves data
    """
    curves = {
        'steps': [],
        'avg_P_total': [],
        'avg_SINR_min': [],
        'success_rate': []
    }

    if parsed_data.get('trajectory_evals'):
        for i, traj in enumerate(parsed_data['trajectory_evals']):
            if 'summary' in traj:
                summary = traj['summary']
                # Assume eval_freq = 10000 (can be made configurable)
                step = (i + 1) * 10000
                curves['steps'].append(step)
                curves['avg_P_total'].append(summary['avg_P_total'])
                curves['avg_SINR_min'].append(summary['avg_SINR_min'])
                curves['success_rate'].append(summary['success_rate'] * 100)

    return curves


def save_parsed_data(parsed_data, output_path):
    """Save parsed data to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(parsed_data, f, indent=2)

    print(f"\nParsed data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Parse training logs from train_rl_with_ae.py')

    parser.add_argument('--log-file', type=str,
                       help='Path to console log file')
    parser.add_argument('--results-dir', type=str,
                       help='Path to results directory containing JSON files')
    parser.add_argument('--output', type=str, default='parsed_results.json',
                       help='Output JSON file path (default: parsed_results.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed summary')

    args = parser.parse_args()

    if not args.log_file and not args.results_dir:
        parser.error("Must specify either --log-file or --results-dir")

    parsed_data = {}

    # Parse console log
    if args.log_file:
        print(f"Parsing console log: {args.log_file}")
        console_data = parse_console_log(args.log_file)
        parsed_data.update(console_data)

    # Parse JSON results
    if args.results_dir:
        print(f"Parsing JSON results from: {args.results_dir}")
        json_data = parse_json_results(args.results_dir)
        # Merge JSON data (console log takes precedence)
        for key in json_data:
            if not parsed_data.get(key):
                parsed_data[key] = json_data[key]

    # Print summary
    if args.verbose:
        print_summary(parsed_data)

    # Save parsed data
    save_parsed_data(parsed_data, args.output)

    # Extract training curves
    curves = extract_training_curves(parsed_data)
    if curves['steps']:
        print(f"\n[Training Curves Summary]")
        print(f"  Evaluations: {len(curves['steps'])}")
        print(f"  Power range: [{min(curves['avg_P_total']):.4f}, {max(curves['avg_P_total']):.4f}] W")
        print(f"  SINR range: [{min(curves['avg_SINR_min']):.2f}, {max(curves['avg_SINR_min']):.2f}] dB")
        print(f"  Success rate range: [{min(curves['success_rate']):.1f}, {max(curves['success_rate']):.1f}]%")

    return 0


if __name__ == '__main__':
    exit(main())
