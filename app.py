import io
import os
import zipfile
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sim_engine import (
    WorkloadConfig, default_architectures, run_comparison, simulate_once,
    workload_model, TASKS
)

st.set_page_config(page_title="5G Multicore Architecture Simulator", page_icon="📡",
                   layout="wide")

st.title("📡 5G Multicore Architecture Simulator")
st.caption("Interactive software model • CPU vs CPU-DSP vs CPU-GPU • Synthetic assumptions, not hardware measurements")

with st.sidebar:
    st.header("Workload configuration")
    antennas = st.select_slider("Number of antennas", options=[8, 16, 32, 64], value=32)
    fft_size = st.selectbox("FFT size", options=[1024, 2048, 4096, 8192], index=2)
    symbols = st.select_slider("OFDM symbols per slot", options=[7, 14], value=14)
    subcarriers = st.selectbox("Active subcarriers", options=[1200, 1800, 3276], index=2)
    pilot_symbols = st.slider("Pilot symbols", min_value=1, max_value=8, value=4)
    resource_blocks = st.slider("Resource blocks", min_value=25, max_value=273, value=273, step=1)
    users = st.slider("Scheduled users", min_value=1, max_value=64, value=32)
    tb_bits = st.select_slider("Transport block size (bits)", options=[50_000, 100_000, 200_000, 400_000], value=200_000)
    code_rate = st.slider("Code rate", min_value=0.25, max_value=0.9, value=0.5, step=0.05)
    deadline_ms = st.slider("Processing deadline (ms)", min_value=0.25, max_value=2.0, value=1.0, step=0.05)
    mc_runs = st.select_slider("Monte Carlo runs", options=[50, 100, 200, 500], value=200)
    jitter = st.slider("Run-to-run variation (σ)", min_value=0.0, max_value=0.25, value=0.10, step=0.01)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=2026, step=1)

    st.divider()
    st.caption("Architecture parameters are editable in sim_engine.py. Defaults are illustrative synthetic assumptions.")

cfg = WorkloadConfig(
    antennas=antennas, fft_size=fft_size, symbols_per_slot=symbols,
    subcarriers=subcarriers, pilot_symbols=pilot_symbols,
    resource_blocks=resource_blocks, users=users,
    transport_block_bits=tb_bits, code_rate=code_rate,
    deadline_ms=deadline_ms, monte_carlo_runs=mc_runs,
    jitter_sigma=jitter, seed=int(seed),
)

if "results" not in st.session_state:
    st.session_state.results = None

top1, top2 = st.columns([1, 4])
with top1:
    run_clicked = st.button("▶ Run simulation", type="primary", use_container_width=True)
with top2:
    st.info("Run the model to calculate task execution, communication costs, latency distributions, throughput and energy estimates.")

if run_clicked:
    with st.spinner("Simulating the three architectures…"):
        st.session_state.results = run_comparison(cfg)

if st.session_state.results is None:
    st.markdown("### What this simulator models")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Homogeneous CPU**\n\nIdentical CPU cores execute all five workloads.")
    c2.markdown("**CPU-DSP**\n\nCPU handles control-oriented work; DSP accelerates signal-processing workloads.")
    c3.markdown("**CPU-GPU**\n\nCPU dispatches parallel workloads to GPU compute units, including modeled transfer overhead.")
    st.warning("This is a simplified software-level model. It is not a 5G waveform simulator, a cycle-accurate emulator, or a real hardware benchmark.")
    st.stop()

summary, task_tables, distributions = st.session_state.results

st.subheader("Architecture comparison")
st.dataframe(summary.style.format({
    "Nominal latency (ms)": "{:.3f}",
    "Throughput (Mbps)": "{:.1f}",
    "Nominal energy (mJ)": "{:.2f}",
    "Mean latency (ms)": "{:.3f}",
    "P95 latency (ms)": "{:.3f}",
    "P99 latency (ms)": "{:.3f}",
    "Deadline misses (%)": "{:.1f}",
    "Mean energy (mJ)": "{:.2f}",
    "Mean throughput (Mbps)": "{:.1f}",
}), use_container_width=True, hide_index=True)

best_idx = summary["Nominal latency (ms)"].idxmin()
best = summary.loc[best_idx]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Lowest modeled latency", str(best["Architecture"]), f'{best["Nominal latency (ms)"]:.3f} ms')
m2.metric("Highest nominal throughput", f'{summary["Throughput (Mbps)"].max():.1f} Mbps')
m3.metric("Lowest nominal energy", f'{summary.loc[summary["Nominal energy (mJ)"].idxmin(), "Architecture"]}',
          f'{summary["Nominal energy (mJ)"].min():.2f} mJ')
m4.metric("Deadline", f"{cfg.deadline_ms:.2f} ms")

st.divider()
left, right = st.columns(2)
with left:
    fig = px.bar(summary, x="Architecture", y="Nominal latency (ms)", color="Architecture",
                 title="Nominal end-to-end modeled latency")
    fig.add_hline(y=cfg.deadline_ms, line_dash="dash", line_color="red",
                  annotation_text="Deadline")
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.bar(summary, x="Architecture", y="Throughput (Mbps)", color="Architecture",
                 title="Modeled throughput")
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    fig = px.bar(summary, x="Architecture", y="Nominal energy (mJ)", color="Architecture",
                 title="Nominal estimated energy per modeled slot")
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.bar(summary, x="Architecture", y="Deadline misses (%)", color="Architecture",
                 title="Monte Carlo deadline-miss rate")
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Task-level execution and communication")
arch_names = list(task_tables.keys())
selected_arch = st.selectbox("Inspect architecture", arch_names)
task_df = task_tables[selected_arch].copy()
task_long = task_df.melt(id_vars=["task"], value_vars=["compute_ms", "memory_ms", "transfer_ms", "sync_ms"],
                         var_name="Component", value_name="Time (ms)")
fig = px.bar(task_long, x="task", y="Time (ms)", color="Component", barmode="stack",
             title=f"Modeled task-time components — {selected_arch}")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(task_df[["task", "path", "compute_ms", "memory_ms", "transfer_ms", "sync_ms",
                      "latency_ms", "energy_mJ"]].style.format({
    "compute_ms": "{:.4f}", "memory_ms": "{:.4f}", "transfer_ms": "{:.4f}",
    "sync_ms": "{:.4f}", "latency_ms": "{:.4f}", "energy_mJ": "{:.3f}",
}), use_container_width=True, hide_index=True)

st.subheader("Latency distribution")
dist_rows = []
for name, values in distributions.items():
    dist_rows.extend([{"Architecture": name, "Latency (ms)": float(v)} for v in values])
dist_df = pd.DataFrame(dist_rows)
fig = px.histogram(dist_df, x="Latency (ms)", color="Architecture", barmode="overlay",
                   opacity=0.65, nbins=45, title="Monte Carlo latency distributions")
fig.add_vline(x=cfg.deadline_ms, line_dash="dash", line_color="red",
              annotation_text="Deadline")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Workload assumptions")
ops, mem, output = workload_model(cfg)
work_df = pd.DataFrame({
    "Task": TASKS,
    "Estimated operation proxy": [ops[t] for t in TASKS],
    "Estimated memory traffic (bytes)": [mem[t] for t in TASKS],
    "Estimated output bytes": [output[t] for t in TASKS],
})
st.dataframe(work_df, use_container_width=True, hide_index=True)

st.caption("Interpretation: the figures are generated by this model's assumptions. They should be described as simulated estimates, not measured hardware performance.")

csv_bytes = summary.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download comparison CSV", csv_bytes, file_name="5g_architecture_comparison.csv",
                   mime="text/csv")
task_csv = pd.concat([df.assign(Architecture=name) for name, df in task_tables.items()], ignore_index=True)
st.download_button("⬇ Download task breakdown CSV", task_csv.to_csv(index=False).encode("utf-8"),
                   file_name="5g_task_breakdown.csv", mime="text/csv")
