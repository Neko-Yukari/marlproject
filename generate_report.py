"""
Auto-Report Generator for Cross-Scale MEC Experiments.

Generates comprehensive comparison report after all experiments complete.
"""
import json
import time
from pathlib import Path
from datetime import datetime


def load_results():
    """Load all experiment results."""
    results = {}
    
    # HyperNetwork results
    hypernet_file = Path("results/hypernetwork/results.json")
    if hypernet_file.exists():
        with open(hypernet_file) as f:
            results['hypernetwork'] = json.load(f)
    
    # Baseline results
    baseline_dir = Path("results/baseline")
    if baseline_dir.exists():
        results['baseline'] = {}
        for f in baseline_dir.glob("*_results.json"):
            config_name = f.stem.replace("_results", "")
            with open(f) as fp:
                results['baseline'][config_name] = json.load(fp)
    
    # GNN results (if available)
    gnn_file = Path("results/gnn_simple/history.json")
    if gnn_file.exists():
        with open(gnn_file) as f:
            results['gnn'] = json.load(f)
    
    return results


def generate_report():
    """Generate comprehensive comparison report."""
    results = load_results()
    
    report = []
    report.append("=" * 80)
    report.append("CROSS-SCALE MEC OFFLOADING - EXPERIMENT REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    # Summary
    report.append("\n## EXECUTIVE SUMMARY\n")
    
    if 'hypernetwork' in results:
        report.append("### HyperNetwork (Cross-Scale)\n")
        final = results['hypernetwork'].get('final_results', {})
        for config, metrics in final.items():
            report.append(f"- {config}: Cost={metrics.get('cost', 'N/A'):.4f}, Comp={metrics.get('comp', 0):.1%}")
    
    if 'baseline' in results:
        report.append("\n### Baseline IPPO (Independent Models)\n")
        for config, data in results['baseline'].items():
            report.append(f"- {config}: Cost={data.get('final_cost', 'N/A'):.4f}, Comp={data.get('final_comp', 0):.1%}")
    
    # Comparison table
    report.append("\n## DETAILED COMPARISON\n")
    report.append("| Method | 2ES-3MD | 2ES-5MD | 3ES-7MD | Cross-Scale? |")
    report.append("|--------|---------|---------|---------|-------------|")
    
    if 'hypernetwork' in results:
        final = results['hypernetwork'].get('final_results', {})
        c1 = f"{final.get('2ES-3MD', {}).get('cost', 0):.4f}"
        c2 = f"{final.get('2ES-5MD', {}).get('cost', 0):.4f}"
        c3 = f"{final.get('3ES-7MD', {}).get('cost', 0):.4f}"
        report.append(f"| HyperNetwork | {c1} | {c2} | {c3} | YES |")
    
    if 'baseline' in results:
        baseline = results['baseline']
        c1 = f"{baseline.get('2ES-3MD', {}).get('final_cost', 0):.4f}"
        c2 = f"{baseline.get('2ES-5MD', {}).get('final_cost', 0):.4f}"
        c3 = f"{baseline.get('3ES-7MD', {}).get('final_cost', 0):.4f}"
        report.append(f"| Baseline IPPO | {c1} | {c2} | {c3} | NO |")
    
    # Analysis
    report.append("\n## ANALYSIS\n")
    
    if 'hypernetwork' in results and 'baseline' in results:
        report.append("### Performance Gap\n")
        hyper_final = results['hypernetwork'].get('final_results', {})
        baseline_final = results['baseline']
        
        for config in ['2ES-3MD', '2ES-5MD', '3ES-7MD']:
            h_cost = hyper_final.get(config, {}).get('cost', 0)
            b_cost = baseline_final.get(config, {}).get('final_cost', 0)
            if h_cost > 0 and b_cost > 0:
                gap = ((h_cost - b_cost) / b_cost) * 100
                report.append(f"- {config}: HyperNetwork is {gap:.1f}% {'worse' if gap > 0 else 'better'} than baseline")
    
    report.append("\n### Cross-Scale Generalization\n")
    if 'hypernetwork' in results:
        final = results['hypernetwork'].get('final_results', {})
        unseen = [k for k in final.keys() if 'unseen' in k.lower() or k not in ['2ES-3MD', '2ES-5MD', '3ES-7MD']]
        if unseen:
            report.append("Unseen configurations tested:")
            for config in unseen:
                metrics = final[config]
                report.append(f"- {config}: Cost={metrics.get('cost', 0):.4f}, Comp={metrics.get('comp', 0):.1%}")
        else:
            report.append("No unseen configurations tested yet.")
    
    # Recommendations
    report.append("\n## RECOMMENDATIONS\n")
    report.append("1. If HyperNetwork cost is within 10% of baseline: Use HyperNetwork for deployment flexibility")
    report.append("2. If gap is 10-30%: Consider tuning HyperNetwork (lr, hidden_dim, architecture)")
    report.append("3. If gap > 30%: HyperNetwork needs significant redesign or this task may not benefit from cross-scale learning")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def save_report(output_file="results/FINAL_REPORT.txt"):
    """Generate and save report."""
    report = generate_report()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)
    print(report)
    return report


if __name__ == "__main__":
    save_report()
