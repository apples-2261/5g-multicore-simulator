"""
5G Multicore Architecture Simulation Engine
===========================================

A transparent, synthetic performance model for comparing:
  1. Homogeneous multicore CPU
  2. Heterogeneous CPU-DSP
  3. CPU-GPU

This is a software-level analytical/discrete-event approximation, NOT a
cycle-accurate hardware simulator and NOT a measurement of physical hardware.

Model assumptions and units are documented in README.md.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd


TASKS = ["FFT", "Channel estimation", "Beamforming", "Scheduling", "LDPC decoding"]


@dataclass
class WorkloadConfig:
    antennas: int = 32
    fft_size: int = 4096
    symbols_per_slot: int = 14
    subcarriers: int = 3276
    pilot_symbols: int = 4
    resource_blocks: int = 273
    users: int = 32
    transport_block_bits: int = 200_000
    code_rate: float = 0.5
    ldpc_iterations: int = 5
    slot_duration_ms: float = 0.5
    deadline_ms: float = 1.0
    monte_carlo_runs: int = 200
    jitter_sigma: float = 0.10
    seed: int = 2026


@dataclass
class ArchitectureConfig:
    name: str
    cpu_cores: int
    cpu_freq_ghz: float
    cpu_simd_flops_cycle: float
    cpu_dense_efficiency: float
    cpu_scalar_ipc: float
    cpu_scalar_efficiency: float
    cpu_memory_gbs: float
    cpu_dram_fraction: float
    cpu_power_w: float
    cpu_idle_power_w: float
    cpu_sync_us: float
    accelerator_units: int = 0
    accelerator_freq_ghz: float = 1.0
    accelerator_ops_cycle: float = 1.0
    accelerator_efficiency: float = 0.5
    accelerator_memory_gbs: float = 0.0
    link_bandwidth_gbs: float = 0.0
    link_latency_us: float = 0.0
    transfer_jitter: float = 0.0
    accelerator_power_w: float = 0.0
    accelerator_idle_power_w: float = 0.0


def default_architectures() -> Dict[str, ArchitectureConfig]:
    # All values are synthetic assumptions, chosen to demonstrate trade-offs.
    # Do not treat them as specifications for a particular commercial product.
    return {
        "Homogeneous CPU": ArchitectureConfig(
            name="Homogeneous CPU", cpu_cores=32, cpu_freq_ghz=2.6,
            cpu_simd_flops_cycle=32, cpu_dense_efficiency=0.30,
            cpu_scalar_ipc=1.5, cpu_scalar_efficiency=0.70,
            cpu_memory_gbs=60, cpu_dram_fraction=0.30, cpu_power_w=250,
            cpu_idle_power_w=60, cpu_sync_us=5,
        ),
        "CPU-DSP": ArchitectureConfig(
            name="CPU-DSP", cpu_cores=16, cpu_freq_ghz=2.4,
            cpu_simd_flops_cycle=16, cpu_dense_efficiency=0.25,
            cpu_scalar_ipc=1.5, cpu_scalar_efficiency=0.65,
            cpu_memory_gbs=40, cpu_dram_fraction=0.25, cpu_power_w=135,
            cpu_idle_power_w=35, cpu_sync_us=4,
            accelerator_units=8, accelerator_freq_ghz=1.2,
            accelerator_ops_cycle=16, accelerator_efficiency=0.65,
            accelerator_memory_gbs=80, link_bandwidth_gbs=25,
            link_latency_us=8, transfer_jitter=0.05,
            accelerator_power_w=55, accelerator_idle_power_w=12,
        ),
        "CPU-GPU": ArchitectureConfig(
            name="CPU-GPU", cpu_cores=12, cpu_freq_ghz=3.0,
            cpu_simd_flops_cycle=16, cpu_dense_efficiency=0.25,
            cpu_scalar_ipc=1.6, cpu_scalar_efficiency=0.70,
            cpu_memory_gbs=45, cpu_dram_fraction=0.20, cpu_power_w=95,
            cpu_idle_power_w=25, cpu_sync_us=4,
            accelerator_units=64, accelerator_freq_ghz=1.4,
            accelerator_ops_cycle=32, accelerator_efficiency=0.45,
            accelerator_memory_gbs=600, link_bandwidth_gbs=24,
            link_latency_us=10, transfer_jitter=0.08,
            accelerator_power_w=180, accelerator_idle_power_w=25,
        ),
    }


def workload_model(cfg: WorkloadConfig) -> Tuple[dict, dict, dict]:
    """Return operation counts, memory traffic bytes, and output bytes per slot."""
    n_ant = max(1, int(cfg.antennas))
    n_lay = max(1, n_ant // 8)
    coded_bits = cfg.transport_block_bits / max(cfg.code_rate, 0.01)
    # The following are simplified operation-count proxies, not a complete NR PHY implementation.
    ops = {
        "FFT": 5 * cfg.fft_size * math.log2(cfg.fft_size) * n_ant * cfg.symbols_per_slot,
        "Channel estimation": 8 * cfg.subcarriers * n_ant * n_lay * cfg.pilot_symbols,
        "Beamforming": (8 * n_ant * n_lay + 8 * n_lay**3) * cfg.subcarriers * cfg.symbols_per_slot,
        "Scheduling": 20 * cfg.resource_blocks * cfg.users,
        "LDPC decoding": coded_bits * 5 * cfg.ldpc_iterations,
    }
    b_complex = 4  # assumed 16-bit I + 16-bit Q representation
    mem = {
        "FFT": 2 * n_ant * cfg.symbols_per_slot * cfg.fft_size * b_complex,
        "Channel estimation": 2 * cfg.subcarriers * n_ant * cfg.pilot_symbols * b_complex,
        "Beamforming": cfg.subcarriers * n_ant * n_lay * b_complex
                      + 2 * cfg.subcarriers * cfg.symbols_per_slot * n_lay * b_complex,
        "Scheduling": cfg.resource_blocks * cfg.users * 64,
        "LDPC decoding": 2 * coded_bits * cfg.ldpc_iterations,
    }
    output = {
        "FFT": n_ant * cfg.symbols_per_slot * cfg.subcarriers * b_complex,
        "Channel estimation": cfg.subcarriers * n_ant * n_lay * b_complex,
        "Beamforming": cfg.subcarriers * cfg.symbols_per_slot * n_lay * b_complex,
        "Scheduling": cfg.resource_blocks * 8,
        "LDPC decoding": cfg.transport_block_bits / 8,
    }
    return ops, mem, output


# DSP/GPU assignment policy is explicit and editable here.
ACCELERATOR_TASKS = {
    "CPU-DSP": {"FFT", "Channel estimation", "Beamforming", "LDPC decoding"},
    "CPU-GPU": {"FFT", "Channel estimation", "Beamforming", "LDPC decoding"},
}


def _cpu_rate(task: str, arch: ArchitectureConfig) -> float:
    if task in {"FFT", "Channel estimation", "Beamforming"}:
        return arch.cpu_freq_ghz * 1e9 * arch.cpu_simd_flops_cycle * arch.cpu_dense_efficiency
    return arch.cpu_freq_ghz * 1e9 * arch.cpu_scalar_ipc * arch.cpu_scalar_efficiency


def _accelerator_rate(arch: ArchitectureConfig) -> float:
    return (arch.accelerator_units * arch.accelerator_freq_ghz * 1e9
            * arch.accelerator_ops_cycle * arch.accelerator_efficiency)


def _chunk_makespan(ops: float, rate: float, units: int, rng: np.random.Generator,
                    chunks_per_unit: int = 8, imbalance: float = 20.0) -> float:
    """List-scheduling approximation for uneven work chunks."""
    units = max(1, int(units))
    n_chunks = max(units, units * chunks_per_unit)
    shares = rng.dirichlet(np.full(n_chunks, imbalance))
    chunk_times = shares * max(0.0, ops) / max(rate, 1.0)
    free = [(0.0, i) for i in range(units)]
    import heapq
    heapq.heapify(free)
    for idx in sorted(range(n_chunks), key=lambda i: -chunk_times[i]):
        t_free, unit = heapq.heappop(free)
        heapq.heappush(free, (t_free + chunk_times[idx], unit))
    return max(t for t, _ in free)


def _simulate_task(task: str, ops: float, mem_bytes: float, output_bytes: float,
                   cfg: WorkloadConfig, arch: ArchitectureConfig,
                   rng: np.random.Generator) -> dict:
    is_accel = arch.name in ACCELERATOR_TASKS and task in ACCELERATOR_TASKS[arch.name]
    if arch.name == "Homogeneous CPU":
        is_accel = False

    if is_accel:
        rate = _accelerator_rate(arch)
        compute_s = _chunk_makespan(ops, rate, arch.accelerator_units, rng)
        mem_s = arch.cpu_dram_fraction * mem_bytes / max(arch.accelerator_memory_gbs * 1e9, 1)
        transfer_bytes = mem_bytes + output_bytes
        transfer_s = (transfer_bytes / max(arch.link_bandwidth_gbs * 1e9, 1)
                      + arch.link_latency_us * 1e-6)
        if arch.transfer_jitter:
            transfer_s *= float(np.exp(rng.normal(0, arch.transfer_jitter)
                                       - 0.5 * arch.transfer_jitter**2))
        sync_s = arch.cpu_sync_us * 1e-6
        # A simple sequential model: transfer + max(compute, accelerator-memory) + sync.
        task_s = transfer_s + max(compute_s, mem_s) + sync_s
        power_w = arch.cpu_idle_power_w + arch.accelerator_power_w + arch.cpu_power_w * 0.15
        path = "Accelerator"
    else:
        rate = _cpu_rate(task, arch)
        compute_s = _chunk_makespan(ops, rate, arch.cpu_cores, rng)
        mem_s = arch.cpu_dram_fraction * mem_bytes / max(arch.cpu_memory_gbs * 1e9, 1)
        transfer_s = 0.0
        sync_s = arch.cpu_sync_us * 1e-6
        task_s = max(compute_s, mem_s) + sync_s
        power_w = arch.cpu_power_w
        path = "CPU"

    return {
        "task": task, "path": path, "compute_ms": compute_s * 1e3,
        "memory_ms": mem_s * 1e3, "transfer_ms": transfer_s * 1e3,
        "sync_ms": sync_s * 1e3, "latency_ms": task_s * 1e3,
        "energy_mJ": power_w * task_s * 1e3,
        "ops": ops, "memory_bytes": mem_bytes, "transfer_bytes": (mem_bytes + output_bytes if is_accel else 0),
    }


def simulate_once(cfg: WorkloadConfig, arch: ArchitectureConfig, seed: int | None = None) -> dict:
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    ops, mem, output = workload_model(cfg)
    task_rows = []
    for task in TASKS:
        row = _simulate_task(task, ops[task], mem[task], output[task], cfg, arch, rng)
        task_rows.append(row)
    task_df = pd.DataFrame(task_rows)
    latency_ms = float(task_df["latency_ms"].sum())
    energy_mJ = float(task_df["energy_mJ"].sum())
    throughput_mbps = cfg.transport_block_bits / (latency_ms / 1000) / 1e6
    return {
        "architecture": arch.name, "latency_ms": latency_ms,
        "throughput_mbps": throughput_mbps, "energy_mJ": energy_mJ,
        "deadline_ms": cfg.deadline_ms,
        "deadline_met": latency_ms <= cfg.deadline_ms,
        "task_results": task_df,
    }


def run_monte_carlo(cfg: WorkloadConfig, arch: ArchitectureConfig) -> dict:
    rng = np.random.default_rng(cfg.seed + 91)
    latencies = []
    energies = []
    for i in range(max(1, int(cfg.monte_carlo_runs))):
        seed = int(rng.integers(0, 2**31 - 1))
        result = simulate_once(cfg, arch, seed=seed)
        # Run-level variation models abstract variation in work size/system conditions.
        jitter = float(np.exp(rng.normal(0, cfg.jitter_sigma)
                              - 0.5 * cfg.jitter_sigma**2))
        latencies.append(result["latency_ms"] * jitter)
        energies.append(result["energy_mJ"] * jitter)
    lat = np.asarray(latencies)
    energy = np.asarray(energies)
    return {
        "architecture": arch.name,
        "mean_latency_ms": float(lat.mean()),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "p99_latency_ms": float(np.percentile(lat, 99)),
        "deadline_miss_pct": float(np.mean(lat > cfg.deadline_ms) * 100),
        "mean_energy_mJ": float(energy.mean()),
        "mean_throughput_mbps": float(cfg.transport_block_bits / (lat.mean() / 1000) / 1e6),
        "latency_samples_ms": lat,
    }


def run_comparison(cfg: WorkloadConfig, architectures: Dict[str, ArchitectureConfig] | None = None):
    architectures = architectures or default_architectures()
    summaries = []
    task_tables = {}
    distributions = {}
    for name, arch in architectures.items():
        nominal = simulate_once(cfg, arch, seed=cfg.seed)
        mc = run_monte_carlo(cfg, arch)
        summaries.append({
            "Architecture": name,
            "Nominal latency (ms)": nominal["latency_ms"],
            "Throughput (Mbps)": nominal["throughput_mbps"],
            "Nominal energy (mJ)": nominal["energy_mJ"],
            "Mean latency (ms)": mc["mean_latency_ms"],
            "P95 latency (ms)": mc["p95_latency_ms"],
            "P99 latency (ms)": mc["p99_latency_ms"],
            "Deadline misses (%)": mc["deadline_miss_pct"],
            "Mean energy (mJ)": mc["mean_energy_mJ"],
            "Mean throughput (Mbps)": mc["mean_throughput_mbps"],
            "Deadline met nominally": nominal["deadline_met"],
        })
        task_tables[name] = nominal["task_results"].copy()
        distributions[name] = mc["latency_samples_ms"]
    return pd.DataFrame(summaries), task_tables, distributions


def config_as_dict(cfg):
    return asdict(cfg)
