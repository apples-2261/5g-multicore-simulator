# 5G Multicore Architecture Simulator

An interactive Python simulation project for comparing three modeled architectures:
- Homogeneous multicore CPU
- Heterogeneous CPU-DSP
- CPU-GPU

## Important scope statement

This project is a **synthetic software-level performance model**. It is not:
- a real 5G NR waveform or protocol-stack simulator,
- a cycle-accurate processor simulator,
- a hardware benchmark,
- proof that any commercial architecture will achieve the displayed performance.

All processor capacities, bandwidths, power values, task operation counts, and efficiencies are illustrative assumptions. Use the outputs in an academic report only when clearly labeled as simulated estimates.

## Requirements

Python 3.10+ recommended. Python 3.14 may work if all packages have compatible wheels.

## Install

Open a terminal in this project folder:

```powershell
python -m pip install -r requirements.txt
```

## Run the interactive application

```powershell
python -m streamlit run app.py
```

A browser tab should open at a local address such as `http://localhost:8501`.

In the sidebar, choose workload parameters, then click **Run simulation**.

## Run the engine from the terminal

```powershell
python run_simulation.py
```

It prints a summary and writes CSV results under `results/`.

## Files

- `sim_engine.py` — workload equations, task scheduling approximation, architecture models, Monte Carlo analysis.
- `app.py` — interactive Streamlit interface and Plotly charts.
- `run_simulation.py` — terminal runner and CSV export.
- `requirements.txt` — Python dependencies.

## What is modeled

Five task categories are included:
1. FFT/OFDM
2. Channel estimation
3. Beamforming
4. Scheduling
5. LDPC decoding

The engine estimates per-task compute time, memory-bound time, CPU-DSP/GPU transfer time, synchronization overhead, nominal latency, throughput, energy, latency percentiles, and deadline misses.

## Simplifications and limitations

- Operation counts are proxies, not a full 3GPP-compliant implementation.
- Work is split into chunks and scheduled across abstract processing units.
- Memory and interconnect costs use simplified bandwidth/latency equations.
- The current model treats task stages sequentially and does not model detailed pipeline overlap.
- The accelerator mapping policy is illustrative and is defined in `sim_engine.py`.
- Energy is estimated from assumed power levels multiplied by modeled task time; it is not measured power.
- Throughput is computed from configured transport-block bits divided by modeled latency.
- Results are sensitive to assumptions. Run sensitivity experiments and disclose the assumptions in the report.

## Suggested academic wording

"The results are outputs of a parameterized software model using explicitly stated synthetic assumptions. They are intended to compare trends under a common modeling framework and should not be interpreted as physical-hardware benchmark measurements."

## Architecture assumptions

Edit `default_architectures()` in `sim_engine.py` to adjust core counts, frequencies, compute efficiencies, memory bandwidths, transfer costs, and power assumptions. Keep a record of every change and use identical workload inputs for all three architectures.
