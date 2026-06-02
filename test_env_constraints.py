"""Quick verification of environment constraints."""
import sys; sys.path.insert(0, '.')
from envs.paper_accurate_env_v2 import PaperAccurateEnv

env = PaperAccurateEnv(3, 2, seed=42)

print("="*60)
print("Verifying environment constraints")
print("="*60)

# Test 1: All local
obs, _ = env.reset(seed=42)
actions = {"device_0": 0, "device_1": 0, "device_2": 0}
env.step(actions)
m = env.get_episode_metrics()
print(f"\n1. All LOCAL:  comp={m['completion_rate']:.1%}  cost={m['avg_cost']:.3f}")

# Test 2: All ES2
obs, _ = env.reset(seed=42)
actions = {"device_0": 2, "device_1": 2, "device_2": 2}
env.step(actions)
m = env.get_episode_metrics()
print(f"2. All ES2:    comp={m['completion_rate']:.1%}  cost={m['avg_cost']:.3f}")

# Test 3: Optimal (MD1,MD2→ES2; MD3→ES1)
obs, _ = env.reset(seed=42)
actions = {"device_0": 2, "device_1": 2, "device_2": 1}
env.step(actions)
m = env.get_episode_metrics()
print(f"3. Optimal:    comp={m['completion_rate']:.1%}  cost={m['avg_cost']:.3f}")

# Test 4: Other allocations
for alloc_name, actions in [
    ("MD1→ES1, MD2→ES2, MD3→ES2", {"device_0": 1, "device_1": 2, "device_2": 2}),
    ("MD1→ES2, MD2→ES1, MD3→ES2", {"device_0": 2, "device_1": 1, "device_2": 2}),
]:
    obs, _ = env.reset(seed=42)
    env.step(actions)
    m = env.get_episode_metrics()
    print(f"4. {alloc_name}: comp={m['completion_rate']:.1%} cost={m['avg_cost']:.3f}")
