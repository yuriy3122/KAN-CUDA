#pragma once
#include <string>
#include <vector>

struct ParsedDataset {
    std::vector<std::string> feature_names;
    std::vector<float> X;   // row-major
    std::vector<int> y;
    int rows = 0;
    int cols = 0;
};

ParsedDataset load_hmda_csv(const std::string& path);
