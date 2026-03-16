#include "csv_reader.h"
#include "preprocess.h"
#include "kan_model.h"
#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char** argv) {
    try {
        std::string csv = argc > 1 ? argv[1] : "train-hmda-data.csv";
        int hidden = argc > 2 ? std::atoi(argv[2]) : 16;
        int grid = argc > 3 ? std::atoi(argv[3]) : 8;
        int k = argc > 4 ? std::atoi(argv[4]) : 3;
        int epochs = argc > 5 ? std::atoi(argv[5]) : 20;
        int batch = argc > 6 ? std::atoi(argv[6]) : 512;
        float lr = argc > 7 ? std::atof(argv[7]) : 1e-2f;

        auto ds = load_hmda_csv(csv);
        std::cout << "Loaded rows=" << ds.rows << " cols=" << ds.cols << "\nFeatures:";
        for (auto& n : ds.feature_names) std::cout << ' ' << n;
        std::cout << "\n";

        auto split = make_splits_and_scale(ds.X, ds.y, ds.rows, ds.cols, 0.7f, 0.15f, 5u);
        int tr_rows = (int)split.y_train.size(), va_rows = (int)split.y_val.size(), te_rows = (int)split.y_test.size();
        std::cout << "Train=" << tr_rows << " Val=" << va_rows << " Test=" << te_rows << "\n";

        KANClassifier model(ds.cols, hidden, grid, k);
        float best_val = 0.0f;
        for (int e = 1; e <= epochs; ++e) {
            float loss = model.train_epoch(split.X_train, split.y_train, tr_rows, ds.cols, batch, lr);
            float train_acc = model.evaluate_accuracy(split.X_train, split.y_train, tr_rows, ds.cols, batch);
            float val_acc = model.evaluate_accuracy(split.X_val, split.y_val, va_rows, ds.cols, batch);
            float test_acc = model.evaluate_accuracy(split.X_test, split.y_test, te_rows, ds.cols, batch);
            if (val_acc > best_val) best_val = val_acc;
            std::cout << "epoch=" << e << " loss=" << loss << " train_acc=" << train_acc << " val_acc=" << val_acc << " test_acc=" << test_acc << "\n";
        }
        std::cout << "Best val_acc=" << best_val << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
