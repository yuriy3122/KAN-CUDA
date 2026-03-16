#pragma once
#include <vector>
#include <random>

struct SplitData {
    std::vector<float> X_train, X_val, X_test;
    std::vector<int> y_train, y_val, y_test;
    std::vector<float> minv, maxv;
    int n_features = 0;
};

SplitData make_splits_and_scale(
    const std::vector<float>& X,
    const std::vector<int>& y,
    int rows,
    int cols,
    float train_ratio = 0.7f,
    float val_ratio = 0.15f,
    unsigned seed = 5u);
