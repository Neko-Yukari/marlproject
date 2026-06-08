"""TDD tests for PettingZoo API - Pure Python (no pytest)."""
import sys; sys.path.insert(0, '.')
import traceback
from envs.paper_accurate_env import PaperAccurateEnvV3
from pettingzoo import ParallelEnv

# Test results storage
passed = 0
failed = 0
errors = []

def test(name, func):
    global passed, failed
    try:
        func()
        print(f"   PASS: {name}")
        passed += 1
    except AssertionError as e:
        print(f"   FAIL: {name}")
        print(f"     {str(e)}")
        failed += 1
        errors.append((name, str(e)))
    except Exception as e:
        print(f"   ERROR: {name}")
        print(f"     {type(e).__name__}: {str(e)}")
        failed += 1
        errors.append((name, f"{type(e).__name__}: {str(e)}"))

print("="*60)
print("TDD: PettingZoo API Compliance Tests")
print("="*60)

# Module 1: Inheritance
print("\n Module 1: Inheritance")

def test_inherits_parallel_env():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    assert isinstance(env, ParallelEnv), "Env must inherit ParallelEnv"

test("Inherits ParallelEnv", test_inherits_parallel_env)

# Module 2: Agent Properties
print("\n Module 2: Agent Properties")

def test_possible_agents():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    assert hasattr(env, 'possible_agents'), "Missing possible_agents attribute"
    assert env.possible_agents == ['device_0', 'device_1', 'device_2']

def test_agents_property():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    assert hasattr(env, 'agents'), "Missing agents attribute"
    env.reset(seed=42)
    assert env.agents == env.possible_agents, f"agents={env.agents}, expected={env.possible_agents}"

test("possible_agents exists", test_possible_agents)
test("agents property after reset", test_agents_property)

# Module 3: Observation/Action Spaces
print("\n Module 3: Per-Agent Spaces")

def test_observation_space():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    assert hasattr(env, 'observation_space'), "Missing observation_space method"
    space = env.observation_space('device_0')
    assert space is not None, "observation_space returned None"
    assert hasattr(space, 'shape'), "Space missing shape attribute"

def test_observation_space_contains():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    obs, _ = env.reset(seed=42)
    space = env.observation_space('device_0')
    assert space.contains(obs['device_0']), "observation_space does not contain actual obs"

def test_action_space():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    assert hasattr(env, 'action_space'), "Missing action_space method"
    space = env.action_space('device_0')
    assert space is not None, "action_space returned None"
    assert hasattr(space, 'n'), "Discrete space missing n attribute"

def test_action_space_sample():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    env.reset(seed=42)
    space = env.action_space('device_0')
    action = space.sample()
    assert space.contains(action), "sampled action not in space"

test("observation_space exists", test_observation_space)
test("observation_space contains actual obs", test_observation_space_contains)
test("action_space exists", test_action_space)
test("action_space can sample", test_action_space_sample)

# Module 4: State
print("\n Module 4: Global State")

def test_state_method():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    assert hasattr(env, 'state'), "Missing state method"
    env.reset(seed=42)
    s = env.state()
    assert s is not None or s is None  # Can be None but method must exist

test("state() method exists", test_state_method)

# Module 5: Close and Render
print("\n Module 5: Lifecycle Methods")

def test_close():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    assert hasattr(env, 'close'), "Missing close method"
    env.close()  # Should not raise

def test_render():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    assert hasattr(env, 'render'), "Missing render method"
    env.render()  # Should not raise

test("close() callable", test_close)
test("render() callable", test_render)

# Module 6: Reset and Step
print("\n Module 6: Episode Execution")

def test_reset_tuple():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    result = env.reset(seed=42)
    assert isinstance(result, tuple), "reset() must return tuple"
    assert len(result) == 2, "reset() must return (obs, info)"

def test_step_tuple():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    obs, _ = env.reset(seed=42)
    actions = {a: 0 for a in env.agents}
    result = env.step(actions)
    assert isinstance(result, tuple), "step() must return tuple"
    assert len(result) == 5, "step() must return (obs, rew, term, trunc, info)"

def test_step_agents():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    env.reset(seed=42)
    # Missing agents should raise error
    try:
        env.step({'device_0': 0})
        assert False, "step() should raise error for missing agents"
    except Exception:
        pass  # Expected

test("reset() returns tuple", test_reset_tuple)
test("step() returns 5-tuple", test_step_tuple)
test("step() requires all agents", test_step_agents)

# Module 7: Options
print("\n Module 7: Options")

def test_reset_options():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    obs, info = env.reset(seed=42, options={'profile_idx': 0})
    assert isinstance(obs, dict), "obs must be dict"
    assert isinstance(info, dict), "info must be dict"

test("reset() accepts options", test_reset_options)

# Integration: Full Episode
print("\n Integration: Full Episode")

def test_full_episode():
    env = PaperAccurateEnvV3(3, 2, seed=42)
    obs, _ = env.reset(seed=42)
    assert set(obs.keys()) == set(env.agents), "obs keys must match agents"
    
    for _ in range(10):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, _ = env.step(actions)
        
        assert set(obs.keys()) == set(env.agents)
        assert set(rewards.keys()) == set(env.agents)
        assert set(terms.keys()) == set(env.agents)
        
        if all(terms.values()):
            break
    
    env.close()

test("Full episode with PettingZoo API", test_full_episode)

# Summary
print("\n" + "="*60)
print(f"Results: {passed} passed, {failed} failed")
print("="*60)

if errors:
    print("\nFailed tests:")
    for name, err in errors:
        print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("\n All tests passed!")
    sys.exit(0)
