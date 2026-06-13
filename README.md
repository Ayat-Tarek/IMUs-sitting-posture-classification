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

The raw recording application logs experimental trials inside folders labeled numerically from `1` to `6`. The pipeline dynamically sorts and maps these directories alphabetically/numerically using the `LABEL_TO_ID` configuration.

Below is the definitive biological mapping of what each numerical folder contains:

| Folder Name | Internal Class ID | Posture Label | Biomechanical Context & Description |
| --- | --- | --- | --- |
| **`1`** | `0` | `Standard Sitting` | Neutral upright seated position; baseline spinal load. |
| **`2`** | `1` | `Slouching` | Thoracic/lumbar flexion; forward kyphotic spine drop. |
| **`3`** | `2` | `Forward Bending` | Deep trunk flexion from hips; intense sagittal spinal displacement. |
| **`4`** | `3` | `Right Twisting` | Axial spine rotation toward the right lateral plane. |
| **`5`** | `4` | `Left Twisting` | Axial spine rotation toward the left lateral plane. |
| **`6`** | `5` | `Standing` | Upright extension; pelvis neutral; weight balanced. |

---

## 3. Preprocessing & Kinematic Engineering Pipeline

To combat hardware inconsistencies and subject-mounting variances, raw files undergo strict digital signal conditioning before hitting the neural network:

```
[Raw CSV File] ──> [Bidirectional Interpolation] ──> [DC Bias Removal]
                         │
                         ▼
[Zero-Phase Low-Pass Filter (3Hz)] <── [Hampel Outlier Suppression]
                         │
                         ▼
[Spatial Coordinate Realignment via Quaternions] ──> [Targeted Feature Slicing]

```

1. **Data Cleaning & Imputation:** Replaces data gaps caused by transient Bluetooth packet loss via linear interpolation, and clamps extreme floating errors.
2. **Signal Conditioning:** Removes DC baseline offset biases and routes data through a non-linear **Hampel Filter** to kill sudden outlier spikes, followed by a zero-phase **Butterworth Low-pass Filter (3.0 Hz)** to capture smooth human movement while blocking high-frequency device vibration.
3. **Spatial Coordinate Realignment:** Uses 4D Quaternions to mathematically rotate local, device-relative coordinates into an absolute, global frame of reference. This neutralizes errors caused by slightly misaligned Velcro straps between subjects.
4. **Targeted Feature Selection:** The system slices out an optimal **24-channel array** (3 Acceleration channels + 3 Euler angles [Yaw, Pitch, Roll] across the 4 physical sensors).

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

### Key Framework Highlights

* **SensorDropout Regularization:** A custom hardware-level regularization layer that completely drops an entire random sensor block with a 25% probability during training batches. This stops the model from leaning lazily on a single dominant node.
* **Hierarchical Attention:** Features two distinct attention sub-networks. *Feature-level attention* weights the most expressive traits inside a single device stream, while *Sensor-level attention* dynamically gauges which anatomical area is providing the most critical movement data at any given millisecond.
* **Leave-One-Subject-Out (LOSO) Cross-Validation:** Evaluates generalizability by holding out an entire individual's data cohort systematically as the test set. This completely eliminates window-shuffling data leakage.
* **Log-Likelihood Window Fusion:** Smooths erratic, instantaneous window classifications into a single, cohesive trial verdict using a stable logarithmic majority vote.
