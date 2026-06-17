def plot_experiment2():
    # Final evals per config. 3ES-7MD GNN uses latest ES-aware result;
    # 2ES configs reuse earlier index-based GNN data as no ES-aware models were trained there.
    data = {
        '2ES-3MD': {
            'Standard-MLP': {'cost': 0.4720, 'std': 0.0094, 'comp': 0.667},
            'GNN': {'cost': 0.4808, 'std': 0.0312, 'comp': 0.667},
            'HyperNet': {'cost': 0.4808, 'std': 0.0453, 'comp': 0.667},
        },
        '2ES-5MD': {
            'Standard-MLP': {'cost': 0.4120, 'std': 0.0111, 'comp': 0.880},
            'GNN': {'cost': 0.4112, 'std': 0.0142, 'comp': 0.900},
            'HyperNet': {'cost': 0.8588, 'std': 0.1561, 'comp': 0.840},
        },
        '3ES-7MD': {
            'Standard-MLP': {'cost': 0.4270, 'std': 0.0129, 'comp': 0.857},
            'GNN': {'cost': 0.4255, 'std': 0.0118, 'comp': 0.860},
            'HyperNet': {'cost': 0.4344, 'std': 0.0189, 'comp': 0.871},
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    configs = ['2ES-3MD', '2ES-5MD', '3ES-7MD']
    x = np.arange(len(configs))
    width = 0.25
    models = ['Standard-MLP', 'GNN', 'HyperNet']
    colors = {'Standard-MLP': '#E55039', 'GNN': '#4A69BD', 'HyperNet': '#78E08F'}

    for i, model in enumerate(models):
        costs = [data[c][model]['cost'] for c in configs]
        stds = [data[c][model]['std'] for c in configs]
        bars = axes[0].bar(x + (i - 1) * width, costs, width, yerr=stds,
                           label=model, color=colors[model], capsize=3)
        for bar, val in zip(bars, costs):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs)
    axes[0].set_ylabel('Average Cost')
    axes[0].set_title('Final Evaluation: Average Cost', fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)

    for i, model in enumerate(models):
        comps = [data[c][model]['comp'] * 100 for c in configs]
        bars = axes[1].bar(x + (i - 1) * width, comps, width, label=model, color=colors[model])
        for bar, val in zip(bars, comps):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                         f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs)
    axes[1].set_ylabel('Completion Rate (%)')
    axes[1].set_title('Final Evaluation: Completion Rate', fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 105])

    fig.suptitle('Experiment 2: Final Evaluation on Training Configurations\n(each model trained and tested on the same configuration; 100 unseen episodes)',
                 fontweight='bold', fontsize=11)
    fig.text(0.5, 0.01, 'Note: GNN results on 3ES-7MD use the ES-aware policy head; 2ES configs use the earlier index-based GNN.',
             ha='center', fontsize=8, style='italic')
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(OUT_DIR / 'exp2_final_evaluation.png', bbox_inches='tight')
    print(f'Saved: {OUT_DIR / "exp2_final_evaluation.png"}')

