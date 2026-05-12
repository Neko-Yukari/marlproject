"""
Helper functions for the MARL edge offloading environment.
Includes delay/energy computation, task generation, and fairness metrics.
"""

import numpy as np
from .task_device import Task, Device


def create_task(task_id: int,
                data_range: tuple = (1e5, 1e6),
                cpu_per_bit: float = 1000.0,
                latency_range: tuple = (0.1, 0.5)) -> Task:
    """
    Create a single computation task with random parameters.

    Args:
        task_id: Unique task identifier
        data_range: (min, max) data size in bits
        cpu_per_bit: CPU cycles required per bit
        latency_range: (min, max) max tolerable latency in seconds

    Returns:
        A Task object with randomized data size and latency constraint.
    """
    return Task(
        task_id=task_id,
        data_size=np.random.uniform(*data_range),
        cpu_cycles_per_bit=cpu_per_bit,
        max_latency=np.random.uniform(*latency_range),
    )


def create_task_batch(num_tasks: int, start_id: int = 0,
                      data_range: tuple = (1e5, 1e6),
                      cpu_per_bit: float = 1000.0,
                      latency_range: tuple = (0.1, 0.5)) -> list:
    """Create a batch of tasks."""
    return [create_task(start_id + i, data_range, cpu_per_bit, latency_range)
            for i in range(num_tasks)]


def calculate_transmission_rate(
    tx_power: float,
    channel_gain: float,
    interference: float,
    bandwidth: float,
    noise: float
) -> float:
    """
    Shannon capacity: rate = B * log2(1 + SNR).
    
    Args:
        tx_power: Transmission power (W)
        channel_gain: Channel gain (linear)
        interference: Total interference power (W)
        bandwidth: System bandwidth (Hz)
        noise: Gaussian noise power (W)
    
    Returns:
        Data rate in bits per second.
    """
    snr = (tx_power * channel_gain) / (interference + noise)
    return bandwidth * np.log2(1 + snr)


def calculate_transmission_time(data_bits: float, rate: float) -> float:
    """Transmission time = data / rate (seconds)."""
    if rate <= 0:
        return float('inf')
    return data_bits / rate


def calculate_local_latency(
    data_bits: float,
    cpu_per_bit: float,
    device_cpu: float
) -> float:
    """Local computation latency (seconds)."""
    return data_bits * cpu_per_bit / device_cpu


def calculate_edge_latency(
    data_bits: float,
    cpu_per_bit: float,
    server_cpu: float,
    tx_rate: float,
    queue_load: float,
    startup_time: float = 0.0
) -> float:
    """
    Edge offloading latency = transmission + queue wait + startup + execution.

    Args:
        data_bits: Amount of data to offload (bits)
        cpu_per_bit: CPU cycles per bit
        server_cpu: Server CPU capacity (cycles/s)
        tx_rate: Transmission rate (bps)
        queue_load: Existing queue load on server (CPU cycles)
        startup_time: Initialization overhead for cold start (seconds)
    
    Returns:
        Total edge computation latency in seconds.
    """
    t_transmission = data_bits / tx_rate if tx_rate > 0 else float('inf')
    t_wait = queue_load / server_cpu
    t_execution = data_bits * cpu_per_bit / server_cpu
    return t_transmission + t_wait + startup_time + t_execution


def calculate_local_energy(
    data_bits: float,
    cpu_per_bit: float,
    device_cpu: float,
    energy_coeff: float = 1e-28
) -> float:
    """
    Local computation energy: E = ξ * f² * D * c (CMOS dynamic power).
    
    Args:
        data_bits: Data processed locally (bits)
        cpu_per_bit: CPU cycles per bit
        device_cpu: Device CPU frequency (Hz)
        energy_coeff: Energy efficiency coefficient ξ [33]
    
    Returns:
        Energy in Joules.
    """
    return energy_coeff * (device_cpu ** 2) * data_bits * cpu_per_bit


def calculate_transmission_energy(
    data_bits: float,
    tx_power: float,
    tx_rate: float
) -> float:
    """Transmission energy = power * time = power * data / rate."""
    if tx_rate <= 0:
        return float('inf')
    return tx_power * data_bits / tx_rate


def compute_fairness_index(values: list) -> float:
    """
    Jain's Fairness Index.
    
    J = (Σ x_i)² / (n * Σ x_i²)
    Range: [1/n, 1], where 1 is perfectly fair.
    """
    if not values or sum(values) == 0:
        return 0.0
    n = len(values)
    sum_sq = sum(x ** 2 for x in values)
    sum_val = sum(values)
    return (sum_val ** 2) / (n * sum_sq) if sum_sq > 0 else 0.0
