"""Run full comparison using train_unified.py - 3 configs x 3 networks."""
import subprocess, sys, json, re
from datetime import datetime
from pathlib import Path

COMBINATIONS = [
    # (network, M, E, label)
    ('standard', 3, 2, 'Standard 2ES-3MD'),
    ('gnn',       3, 2, 'GNN 2ES-3MD'),
    ('hyper',     3, 2, 'Hyper 2ES-3MD'),
    ('standard', 5, 2, 'Standard 2ES-5MD'),
    ('gnn',       5, 2, 'GNN 2ES-5MD'),
    ('hyper',     5, 2, 'Hyper 2ES-5MD'),
    ('standard', 7, 3, 'Standard 3ES-7MD'),
    ('gnn',       7, 3, 'GNN 3ES-7MD'),
    ('hyper',     7, 3, 'Hyper 3ES-7MD'),
]

EPISODES = 10000
ALGO = 'ippo'
DEVICE = 'cuda'

log_path = Path('results/comparison_log.txt')
log_path.parent.mkdir(exist_ok=True)

results = []
total = len(COMBINATIONS)

with open(log_path, 'w', encoding='utf-8') as log:
    log.write(f"===== Full Comparison Started: {datetime.now()} =====\n")
    log.write(f"Total: {total} experiments, {EPISODES} episodes each\n\n")
    log.flush()
    
    for idx, (network, M, E, label) in enumerate(COMBINATIONS, 1):
        print(f"\n[{idx}/{total}] {label}")
        log.write(f"\n[{idx}/{total}] {label}\n")
        log.flush()
        
        cmd = [
            sys.executable, 'train_unified.py',
            '--network', network,
            '--algorithm', ALGO,
            '--md', str(M), '--es', str(E),
            '--episodes', str(EPISODES),
            '--device', DEVICE, '--seed', '42'
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                text=True, cwd='.', encoding='gbk', errors='replace')
        
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})")
            log.write(f"  FAILED (exit {result.returncode})\n")
        
        # Parse stdout (small output, no deadlock risk with gbk encoding)
        for line in (result.stdout or '').splitlines():
            if 'Ep ' in line and ('Cost:' in line or 'Avg Cost:' in line):
                log.write(f"  {line}\n")
            if 'Avg Cost:' in line:
                print(f"  {line.strip()}")
        
        log.write(f"  Return code: {result.returncode}\n")
        log.flush()
    
    log.write(f"\n===== Comparison Complete: {datetime.now()} =====")

print(f"\nDone! Log: {log_path}")
