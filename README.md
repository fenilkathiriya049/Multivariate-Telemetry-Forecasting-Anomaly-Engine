# ⚡ Industrial Telemetry Forecasting & Anomaly Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.17+-005CED.svg?style=flat&logo=onnx)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end multivariate time-series forecasting and physical anomaly detection pipeline. This project contrasts recurrent inductive biases (**Stacked Bi-LSTM + Bahdanau Temporal Attention**) against modern transformer architectures (**Channel-Independent PatchTST**) on industrial electricity transformer telemetry (**ETTh1**). 

The system incorporates **Reversible Instance Normalization (RevIN)** to combat non-stationary distribution drift, dynamic residual **EWMA Z-score** thresholding for sensor fault detection, and **ONNX Runtime** optimization for edge serving.

---

## 📌 Key Architectural Takeaways

1. **Local Temporal Grouping via Patching:** Slicing raw sequences into overlapping patches ($P=16, S=8$) extracts local cyclical semantics and reduces self-attention complexity from $\mathcal{O}(L^2)$ to $\mathcal{O}((L/S)^2)$.
2. **Channel Independence (CI):** Treating each multivariate sensor as an independent univariate channel passed through shared transformer weights acts as an effective regularizer, suppressing spurious cross-sensor noise.
3. **RevIN (Reversible Instance Normalization):** Normalizes instance-level mean and variance before the encoder and denormalizes predictions after the head, mitigating covariate shift over long time horizons.

---

## 📊 Experimental Results

### 1. Model Accuracy Benchmark (Lookback $L=96$, Forecast $H=24$)

Evaluated on the held-out test split (final 20% chronological split of `ETTh1`):

| Model Architecture | Trainable Params | Test MSE | Test MAE | Relative MSE Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Stacked Bi-LSTM + Bahdanau Attention** | 200,984 | 10.4186 | 1.8177 | Baseline |
| **PatchTST (Channel-Independent)** | 48,152 | **7.4900** | **1.4382** | **-28.11%** |

### 2. Anomaly Detection Operational Sweep

Residual error dynamic Z-score thresholding evaluated against ground-truth physical fault injections:

| Threshold ($Z$) | Precision | Recall | F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| $Z = 1.5$ | 93.01% | 100.00% | 96.38% | High False Positive Rate |
| $Z = 1.8$ | 96.11% | 100.00% | 98.02% | Balanced |
| **$Z = 2.2$** | **98.86%** | **100.00%** | **99.43%** | **Optimal Operating Point** |
| $Z = 2.8$ | 98.86% | 100.00% | 99.43% | High Confidence Bound |

### 3. Edge Serving Latency & Throughput Benchmark (Batch=1, CPU)

Benchmarked over 200 iterations with warm-up cycles:

| Model Architecture | Runtime Engine | $P_{50}$ Median Latency | $P_{95}$ Tail Latency | Throughput (FPS) |
| :--- | :--- | :--- | :--- | :--- |
| **Bi-LSTM** | PyTorch Native | 1.53 ms | 2.29 ms | 612.6 Hz |
| **Bi-LSTM** | ONNX Runtime | 1.07 ms | 1.62 ms | 932.4 Hz |
| **PatchTST** | PyTorch Native | 0.92 ms | 1.78 ms | 968.0 Hz |
| **PatchTST** | **ONNX Runtime** | **0.66 ms** | **0.71 ms** | **1,518.5 Hz** |

*Key finding:* PatchTST executed in ONNX Runtime delivers **1.62× higher throughput** and a **56% tighter tail latency ($P_{95}$)** than Bi-LSTM ONNX on CPU edge environments.

---

## 📁 Repository Layout

```text
timeseries-benchmark/
├── checkpoints/              # Best-performing model weights (.pt)
├── data/                     # ETTh1 benchmark telemetry (downloaded automatically)
├── onnx_models/              # Exported ONNX graph definitions
├── src/
│   ├── dataset.py            # Causal sliding-window dataset & data loader factory
│   ├── models/
│   │   ├── revin.py          # Reversible Instance Normalization module
│   │   ├── bi_lstm.py        # Bi-LSTM with Bahdanau attention
│   │   └── patch_tst.py      # Channel-Independent PatchTST
│   └── evaluation/
│       ├── metrics.py        # MSE & MAE calculation
│       ├── trainer.py        # Unified training engine with early stopping
│       └── anomaly_detector.py # EWMA dynamic Z-score anomaly detector
├── app.py                    # Streamlit interactive monitoring dashboard
├── benchmark_latency.py      # ONNX export and P50/P95 latency profiler
├── evaluate_anomalies.py     # Fault simulation and threshold calibration script
├── requirements.txt          # Pinned project dependencies
└── train.py                  # End-to-end model training entrypoint

🚀 Quickstart & Reproducibility
1. Environment Setup
git clone [https://github.com/](https://github.com/)<your-username>/timeseries-benchmark.git
cd timeseries-benchmark

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt

2. Train Models from Scratch
Downloads ETTh1.csv automatically, enforces strict chronological splitting (70/10/20), trains both models with early stopping, saves optimal checkpoints, and prints test split metrics:
python train.py

3. Run Anomaly Detection Calibration
Evaluates dynamic residual thresholding against synthetic sensor fault injection:

python evaluate_anomalies.py

4. Export to ONNX & Run Latency Profiling
Exports computational graphs to onnx_models/ and profiles Batch-1 latency:
python benchmark_latency.py

5. Launch the Streamlit Monitoring Dashboard
Launch the interactive web UI to inspect forecasts, compare curves across the 7 sensor channels, and simulate sensor spikes live:
streamlit run app.py

📜 References
PatchTST: Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR 2023.

RevIN: Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2022). Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. ICLR 2022.

ETT Dataset: Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI 2021.