# Deep Learning Posture Classification Pipeline

This repository contains an end-to-end deep learning framework designed to classify human macro-postures using raw streams from a multi-sensor array of four Inertial Measurement Units (IMUs) positioned along the spinal axis (**C7**, **T4**, **T12**, and **L5**).

The architecture leverages parallel 1D Convolutional Neural Networks (1D-CNN) integrated with a hierarchical dual-level attention mechanism (feature-level and sensor-level attention) to deliver robust, subject-independent posture tracking.

---

## 1. Repository Directory Structure

The preprocessing pipeline expects your local directory to be organized as follows:

```text
├── README.md
├── IMUs_Sitting_Posture_Baseline_CNN_Classifier_LOSO.ipynb   # Master pipeline notebook (Cells 1-13)
└── IMU_Posture_Dataset_Wide_format/                          # Root data directory (RAW_ROOT)
    ├── 1/                                                    # Folder containing Backward Bending trials
    │   ├── Sub_1_Trial_1.csv
    │   └── Sub_1_Trial_2.csv
    ├── 2/                                                    # Folder containing Upright trials
    ├── 3/                                                    # Folder containing Slouching trials
    ├── 4/                                                    # Folder containing Forward Bending trials
    ├── 5/                                                    # Folder containing Right Bending trials
    └── 6/                                                    # Folder containing Left Bending trials

```

---


## 2. Dataset Folder Mapping & Posture Definitions

The raw recording hardware application logs experimental trials inside folders labeled numerically from `1` to `6`. The data pipeline dynamically sorts and maps these directories alphabetically and numerically using the internal `LABEL_TO_ID` tracking schema.

Below is the definitive mapping of what each numerical directory contains:

| Folder Name | Posture Label |
| --- |  --- |
| **`1`** |  `Backward Bending` |
| **`2`** |  `Upright` |
| **`3`** |  `Slouching` |
| **`4`** |  `Forward Bending` |
| **`5`** |  `Right Bending` |
| **`6`** |  `Left Bending` |

---

## 3. Preprocessing & Kinematic Engineering Pipeline

To combat ambient electronic noise, anatomical drift, and subject-mounting variations, raw files undergo strict digital signal conditioning before hitting the neural network:

```
[Raw CSV File] ──> [Bidirectional Interpolation] ──> [DC Bias Removal]
                                                         │
                                                         ▼
[Zero-Phase Low-Pass Filter (3Hz)] <── [Hampel Outlier Suppression]
         │
         ▼
[Spatial Coordinate Realignment via Quaternions] ──> [Targeted Feature Slicing]

```

1. **Data Integrity & Imputation:** Replaces data gaps caused by transient wireless Bluetooth packet loss via linear bidirectional interpolation, and forces invalid strings into numerical boundaries.
2. **Signal Conditioning (DSP):** Removes static DC baseline sensor offset biases. Arrays are then routed through a non-linear **Hampel Filter** to suppress sudden spike/impulse artifacts, followed by a zero-phase **Butterworth Low-pass Filter (3.0 Hz)** to preserve smooth macro-postural trajectories while eliminating high-frequency tremors.
3. **Spatial Coordinate Realignment:** Uses 4D Quaternions to mathematically rotate local, device-relative coordinates into an absolute, global earth frame of reference. This neutralizes errors caused by slightly misaligned sensor orientations or shifted Velcro straps across different subjects.
4. **Targeted Feature Selection:** The system slices out an optimal, highly interpretable **24-channel array** (3 Acceleration axes + 3 Biomechanical Euler angles [Yaw, Pitch, Roll] across the 4 physical spinal sensors), discarding high-variance or redundant metrics.

---

## 4. Model Architecture & Evaluation Strategy

```
                          [24-Channel Input Window]
                                      │
                                      ▼
                        [SensorDropout Layer (25%)]
                                      │
             ┌────────────────┬───────┴────────┬────────────────┐
             ▼                ▼                ▼                ▼
         [L5 Stream]     [T4 Stream]      [C7 Stream]     [T12 Stream]
         (6 Channels)    (6 Channels)     (6 Channels)    (6 Channels)
             │                │                │                │
             ▼                ▼                ▼                ▼
         [1D-CNN]         [1D-CNN]         [1D-CNN]         [1D-CNN]
             │                │                │                │
             ▼                ▼                ▼                ▼
        [Feat-Attn]      [Feat-Attn]      [Feat-Attn]      [Feat-Attn]
             │                │                │                │
             └────────────────┼───────┬────────┴────────────────┘
                                      ▼
                           [Sensor-Level Attention]
                                      │
                                      ▼
                        [Global Average Pooling 1D]
                                      │
                                      ▼
                          [Dense Classifier Head]
                                      │
                                      ▼
                         [6-Class Posture Softmax]

```

### Framework Highlights

* **SensorDropout Regularization:** A custom hardware-level regularization layer that completely drops an entire random sensor block with a 25% probability during training batches. This stops the network from over-relying on a single dominant sensor placement.
* **Hierarchical Attention Mechanism:** Features two distinct attention sub-networks. *Feature-level attention* dynamically weights the most expressive traits inside a single device stream, while *Sensor-level attention* gauges which anatomical area is providing the most critical movement data at any given millisecond.
* **Leave-One-Subject-Out (LOSO) Cross-Validation:** Evaluates true user generalizability by holding out an entire individual's data cohort systematically as the independent testing set. This completely eliminates window-shuffling data leakage.
* **Log-Likelihood Window Fusion:** Smooths erratic, instantaneous window classifications into a single, cohesive trial verdict using a stable logarithmic majority vote, yielding robust clinical metric auditing.

```

```
