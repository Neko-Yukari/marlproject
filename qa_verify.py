"""QA verification script for MARL-IPPOAndMore environment."""
import numpy as np
from envs.edge_offload_env import EdgeOffloadEnv
from gymnasium import spaces

def test_basic_env():
    print("=" * 60)
    print("TEST 1: Basic env test (single step, 5 agents)")
    print("=" * 60)
    env = EdgeOffloadEnv(num_devices=5, num_servers=3, max_slots=100, seed=42)
    obs, info = env.reset(seed=42)
    agents = list(obs.keys())
    print(f"Agents listed: {agents}")
    assert len(agents) == 5, f"Expected 5 agents, got {len(agents)}"

    actions = {}
    for a in agents:
        actions[a] = {
            "offload_ratio": np.array([0.5], dtype=np.float32),
            "target_es": 1,
        }
    observations, rewards, terminations, truncations, infos = env.step(actions)
    reward_vals = [rewards[a] for a in agents]
    print(f"Rewards returned: {reward_vals}")
    assert len(reward_vals) == 5, f"Expected 5 rewards, got {len(reward_vals)}"
    # Check rewards are different from each other
    unique_rewards = len(set(round(r, 6) for r in reward_vals))
    print(f"Unique rewards (6dp): {unique_rewards}")
    assert unique_rewards > 1, f"Rewards are all identical: {reward_vals}"
    print("TEST 1: PASS\n")
    return True

def test_diversity():
    print("=" * 60)
    print("TEST 2: Diversity test (10 steps, varying actions)")
    print("=" * 60)
    env = EdgeOffloadEnv(num_devices=5, num_servers=3, max_slots=100, seed=42)
    obs, info = env.reset(seed=42)
    agents = list(obs.keys())
    all_different = True
    for step in range(10):
        actions = {}
        for i, a in enumerate(agents):
            offload_ratio = 0.2 + 0.1 * i
            target_es = i % 3 + 1
            actions[a] = {
                "offload_ratio": np.array([offload_ratio], dtype=np.float32),
                "target_es": target_es,
            }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        reward_vals = [rewards[a] for a in agents]
        unique_rewards = len(set(round(r, 6) for r in reward_vals))
        print(f"Step {step}: rewards={reward_vals}, unique={unique_rewards}")
        if unique_rewards < 2:
            all_different = False
            print(f"  WARNING: All rewards identical at step {step}")
    assert all_different, "Not all steps had differing rewards per agent"
    print("TEST 2: PASS\n")
    return True

def test_pettingzoo_api():
    print("=" * 60)
    print("TEST 5: PettingZoo API check")
    print("=" * 60)
    env = EdgeOffloadEnv(num_devices=5, num_servers=3, max_slots=100, seed=42)
    from pettingzoo import ParallelEnv
    assert isinstance(env, ParallelEnv), "Env does not inherit from ParallelEnv"
    print("Inherits ParallelEnv: PASS")

    assert hasattr(env, 'possible_agents'), "Missing possible_agents"
    print(f"possible_agents: {env.possible_agents}")
    assert len(env.possible_agents) == 5

    assert hasattr(env, 'observation_spaces'), "Missing observation_spaces"
    obs_spaces = env.observation_spaces
    print(f"observation_spaces keys: {list(obs_spaces.keys())}")
    assert isinstance(obs_spaces, dict), "observation_spaces is not a Dict"
    for a, sp in obs_spaces.items():
        assert isinstance(sp, spaces.Box), f"obs space for {a} is not Box"
    print("observation_spaces (Dict of Box): PASS")

    assert hasattr(env, 'action_spaces'), "Missing action_spaces"
    act_spaces = env.action_spaces
    print(f"action_spaces keys: {list(act_spaces.keys())}")
    assert isinstance(act_spaces, dict), "action_spaces is not a Dict"
    for a, sp in act_spaces.items():
        assert isinstance(sp, spaces.Dict), f"action space for {a} is not Dict"
    print("action_spaces (Dict of Dict): PASS")
    print("TEST 5: PASS\n")
    return True

def test_multi_agent_correctness():
    print("=" * 60)
    print("TEST 6: Multi-agent correctness (random actions, 5 steps)")
    print("=" * 60)
    env = EdgeOffloadEnv(num_devices=5, num_servers=3, max_slots=100, seed=42)
    obs, info = env.reset(seed=42)
    agents = list(obs.keys())
    diffs_found = 0
    for step in range(5):
        actions = {}
        for a in agents:
            actions[a] = {
                "offload_ratio": np.array([np.random.rand()], dtype=np.float32),
                "target_es": np.random.randint(0, 4),
            }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        r0 = rewards["device_0"]
        r4 = rewards["device_4"]
        print(f"Step {step}: agent_0={r0:.6f}, agent_4={r4:.6f}, diff={abs(r0-r4):.6f}")
        if abs(r0 - r4) > 1e-9:
            diffs_found += 1
    print(f"Steps with differing agent_0 vs agent_4 rewards: {diffs_found}/5")
    assert diffs_found >= 1, "agent_0 and agent_4 rewards never differed"
    print("TEST 6: PASS\n")
    return True

if __name__ == "__main__":
    results = {}
    try:
        results["basic"] = test_basic_env()
    except Exception as e:
        print(f"TEST 1: FAIL - {e}\n")
        results["basic"] = False

    try:
        results["diversity"] = test_diversity()
    except Exception as e:
        print(f"TEST 2: FAIL - {e}\n")
        results["diversity"] = False

    try:
        results["pettingzoo_api"] = test_pettingzoo_api()
    except Exception as e:
        print(f"TEST 5: FAIL - {e}\n")
        results["pettingzoo_api"] = False

    try:
        results["multi_agent"] = test_multi_agent_correctness()
    except Exception as e:
        print(f"TEST 6: FAIL - {e}\n")
        results["multi_agent"] = False

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
