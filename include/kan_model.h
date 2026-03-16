#pragma once
#include <vector>
#include <string>

struct DeviceArray {
    float* ptr = nullptr;
    int size = 0;
    DeviceArray() = default;
    explicit DeviceArray(int n);
    ~DeviceArray();
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&& other) noexcept;
    DeviceArray& operator=(DeviceArray&& other) noexcept;
    void resize(int n);
    void from_host(const std::vector<float>& v);
    void to_host(std::vector<float>& v) const;
};

struct DeviceIntArray {
    int* ptr = nullptr;
    int size = 0;
    DeviceIntArray() = default;
    explicit DeviceIntArray(int n);
    ~DeviceIntArray();
    DeviceIntArray(const DeviceIntArray&) = delete;
    DeviceIntArray& operator=(const DeviceIntArray&) = delete;
    void resize(int n);
    void from_host(const std::vector<int>& v);
};

struct KANLayer {
    int in_dim = 0;
    int out_dim = 0;
    int grid_intervals = 5;
    int k = 3;
    int n_knots = 0;
    int n_coef = 0;

    DeviceArray grid;
    DeviceArray coef;
    DeviceArray scale_base;
    DeviceArray scale_sp;

    DeviceArray grad_coef;
    DeviceArray grad_scale_base;
    DeviceArray grad_scale_sp;

    KANLayer() = default;
    KANLayer(int in_dim_, int out_dim_, int grid_intervals_, int k_);
    void init(float low = -1.0f, float high = 1.0f, unsigned seed = 42u);
    void forward(const DeviceArray& x, DeviceArray& out, int batch) const;
    void backward(const DeviceArray& grad_out, const DeviceArray& x, DeviceArray& grad_x, int batch);
    void step(float lr, float weight_decay = 0.0f);
    void zero_grads();
};

struct KANClassifier {
    KANLayer l1;
    KANLayer l2;
    int input_dim = 0;
    int hidden_dim = 0;
    int num_classes = 2;

    KANClassifier(int input_dim_, int hidden_dim_, int grid_, int k_);
    float train_epoch(const std::vector<float>& X, const std::vector<int>& y, int rows, int cols, int batch_size, float lr);
    float evaluate_accuracy(const std::vector<float>& X, const std::vector<int>& y, int rows, int cols, int batch_size) const;
};
