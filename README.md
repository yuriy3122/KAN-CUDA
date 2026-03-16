# CUDA KAN credit model (pure C++/CUDA)

This package is a pure C++/CUDA training pipeline derived from the logic in `trainModel.py`.
It does **not** use PyTorch.

## What is implemented

- CSV parsing for `train-hmda-data.csv`
- row skipping for missing values / `Exempt`
- feature selection matching the Python script logic:
  - drop `action_taken`, `loan_amount`, `income`, `loan_term`, `property_value`
  - then take remaining columns slice `[1:19]`
- stratified train/val/test split
- MinMax scaling fit on **train only**
- two-layer KAN classifier:
  - layer1: `[n_features -> hidden]`
  - layer2: `[hidden -> 2]`
- CUDA forward/backward for each KAN layer
- SGD training loop with softmax cross-entropy on CUDA

## File layout

- `include/csv_reader.h`
- `include/preprocess.h`
- `include/kan_cuda_kernels.h`
- `include/kan_model.h`
- `src/csv_reader.cpp`
- `src/preprocess.cpp`
- `src/kan_model.cpp`
- `src/kan_cuda_kernels.cu`
- `src/train.cpp`
- `CMakeLists.txt`

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build . -j
```

## Run

```bash
./kan_credit_train /path/to/train-hmda-data.csv 16 8 3 20 512 0.01
```

Arguments:
1. csv path
2. hidden width (default `16`)
3. grid intervals (default `8`)
4. spline degree k (default `3`, clamped to `<= 3` in this baseline)
5. epochs (default `20`)
6. batch size (default `512`)
7. learning rate (default `0.01`)

## Important notes

This is a **clean baseline CUDA implementation**, not a full feature-parity port of PyKAN.
Not implemented:
- Optuna search
- LBFGS optimizer
- symbolic regression / plotting
- model checkpoint serialization
- ROC/PR plotting

The CUDA kernels are based on the earlier custom KAN kernel and use shared memory in forward for knot grid and coefficient tiles.
Backward is correct in structure but still relies on `atomicAdd`, so it is not the final optimized production version.
