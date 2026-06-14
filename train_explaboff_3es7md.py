"""Train all 3 ExplabOff variants on 3ES-7MD sequentially."""
import subprocess, sys, os

os.environ['PYTHONUNBUFFERED'] = '1'

configs = [
    ("Standard", "configs/explaboff_standard_7md3es.yaml"),
    ("GNN", "configs/explaboff_gnn_7md3es.yaml"),
    ("Hyper", "configs/explaboff_hyper_7md3es.yaml"),
]

print("=" * 60)
print("ExplabOff 3ES-7MD Sequential Training")
print("=" * 60)

for i, (name, cfg) in enumerate(configs):
    print(f"\n[{i+1}/3] Training ExplabOff + {name} ...")
    sys.stdout.flush()
    
    result = subprocess.run(
        [sys.executable, "-u", "train_unified.py", "--config", cfg, "--episodes", "10000"],
        capture_output=False,
        text=True,
        timeout=3600,
    )
    
    if result.returncode == 0:
        print(f"[{i+1}/3] ExplabOff + {name}: SUCCESS (exit {result.returncode})")
    else:
        print(f"[{i+1}/3] ExplabOff + {name}: FAILED (exit {result.returncode})")
    sys.stdout.flush()

print("\n" + "=" * 60)
print("ALL TRAINING COMPLETE")
print("=" * 60)
