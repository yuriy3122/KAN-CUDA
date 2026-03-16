#include "preprocess.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

static void append_rows(const std::vector<float>& X, const std::vector<int>& y, int cols,
                        const std::vector<int>& ids, std::vector<float>& outX, std::vector<int>& outY) {
    outX.reserve(ids.size() * cols);
    outY.reserve(ids.size());
    for (int id : ids) {
        outY.push_back(y[id]);
        const float* row = &X[id * cols];
        outX.insert(outX.end(), row, row + cols);
    }
}

SplitData make_splits_and_scale(const std::vector<float>& X, const std::vector<int>& y,
                                int rows, int cols, float train_ratio, float val_ratio, unsigned seed) {
    if ((int)y.size() != rows) throw std::runtime_error("y size mismatch");
    std::vector<int> idx0, idx1;
    for (int i = 0; i < rows; ++i) (y[i] == 0 ? idx0 : idx1).push_back(i);
    std::mt19937 rng(seed);
    std::shuffle(idx0.begin(), idx0.end(), rng);
    std::shuffle(idx1.begin(), idx1.end(), rng);

    auto split_class = [&](const std::vector<int>& src, std::vector<int>& tr, std::vector<int>& va, std::vector<int>& te) {
        int n = (int)src.size();
        int ntr = (int)(n * train_ratio);
        int nva = (int)(n * val_ratio);
        tr.insert(tr.end(), src.begin(), src.begin() + ntr);
        va.insert(va.end(), src.begin() + ntr, src.begin() + ntr + nva);
        te.insert(te.end(), src.begin() + ntr + nva, src.end());
    };

    std::vector<int> tr, va, te;
    split_class(idx0, tr, va, te);
    split_class(idx1, tr, va, te);
    std::shuffle(tr.begin(), tr.end(), rng);
    std::shuffle(va.begin(), va.end(), rng);
    std::shuffle(te.begin(), te.end(), rng);

    SplitData out;
    out.n_features = cols;
    append_rows(X, y, cols, tr, out.X_train, out.y_train);
    append_rows(X, y, cols, va, out.X_val, out.y_val);
    append_rows(X, y, cols, te, out.X_test, out.y_test);

    out.minv.assign(cols, std::numeric_limits<float>::max());
    out.maxv.assign(cols, std::numeric_limits<float>::lowest());
    for (int r = 0; r < (int)tr.size(); ++r) {
        for (int c = 0; c < cols; ++c) {
            float v = out.X_train[r * cols + c];
            out.minv[c] = std::min(out.minv[c], v);
            out.maxv[c] = std::max(out.maxv[c], v);
        }
    }

    auto scale_block = [&](std::vector<float>& block) {
        int nrows = (int)block.size() / cols;
        for (int r = 0; r < nrows; ++r) {
            for (int c = 0; c < cols; ++c) {
                float mn = out.minv[c], mx = out.maxv[c];
                float& v = block[r * cols + c];
                if (mx > mn) v = (v - mn) / (mx - mn);
                else v = 0.0f;
            }
        }
    };
    scale_block(out.X_train);
    scale_block(out.X_val);
    scale_block(out.X_test);
    return out;
}
