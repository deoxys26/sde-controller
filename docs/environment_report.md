# Environment Report

## 1. Operating System
**Windows**

## 2. Python Version
A recent Python 3 version is installed and active.

## 3. Available CPU and GPU
- **CPU**: Available.
- **GPU**: No NVIDIA GPU detected (`nvidia-smi` failed). PyTorch will run on the CPU.

## 4. Existing Project Files and Repository Structure
The workspace (`c:\Users\Admin\OneDrive\Desktop\DN_proj`) contains:
- `papers/`: Three research paper PDFs (DG-STMTL, Network Traffic Prediction GNN + Anomaly Detection, QoS-based Routing GA in SDN).
- `data/raw/security/unsw_nb15/`: UNSW-NB15 training and testing CSVs.
- `docs/`: Markdown documents.
No existing Python implementation codebase is present.

## 5. Installed Python Packages
- **Available Key Packages**: `torch` (2.12.0), `networkx` (3.6.1), `scikit-learn` (1.4.2), `numpy`, `pandas`.
- **Missing Key Packages**: OpenFlow controller libraries (e.g., `ryu`).

## 6. PyTorch Geometric Compatibility Verification
A dry-run installation (`pip install torch_geometric --dry-run`) was performed. The environment successfully resolved the dependencies and confirmed that `torch-geometric-2.8.0.post1` can be cleanly installed without conflicts. The package has *not* been permanently installed yet, as per Phase 1 strict limits, but compatibility is verified.

## 7. NetworkX and Scikit-learn
- Both are installed and ready for topology representation and ML baselines.

## 8. Mininet and Open vSwitch
- Not natively available on Windows. `wsl -l -v` shows only `docker-desktop` is installed.

## 9. OpenFlow-compatible Controller
- Not installed.

## 10. UNSW-NB15 Dataset Verification
A programmatic verification script was executed against the raw UNSW-NB15 dataset files:
- **Files**: `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` exist.
- **Row Counts**: Train = 175,341 rows | Test = 82,332 rows.
- **Columns**: 45 distinct columns.
- **Target/Label Columns**: Confirmed present (`label` and `attack_cat`).
- **Missing Values**: 0 missing values in both sets.
- **Duplicate Rows**: 0 duplicate rows in the training set.
- **Feature Distribution**: 4 categorical features (`proto`, `service`, `state`, `attack_cat`) and 41 numerical features.
- **Attack Category Distribution (Train)**:
  - Normal: 56,000
  - Generic: 40,000
  - Exploits: 33,393
  - Fuzzers: 18,184
  - DoS: 12,264
  - Reconnaissance: 10,491
  - Analysis: 2,000
  - Backdoor: 1,746
  - Shellcode: 1,133
  - Worms: 130
*Note: The raw CSVs were strictly inspected and have not been modified.*

## 11. Recommended Implementation Path
We will build the custom Python simulator (Phase 3) as the primary network environment. Mininet and SDN controller integration are deferred to Phase 9.
