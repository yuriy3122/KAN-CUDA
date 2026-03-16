#include "csv_reader.h"
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"') in_quotes = !in_quotes;
        else if (c == ',' && !in_quotes) {
            out.push_back(cur);
            cur.clear();
        } else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

static bool bad_cell(const std::string& s) {
    return s.empty() || s == "Exempt" || s == "NA" || s == "NaN";
}

ParsedDataset load_hmda_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open CSV: " + path);

    std::string header_line;
    if (!std::getline(f, header_line)) throw std::runtime_error("Empty CSV");
    auto headers = split_csv_line(header_line);

    std::unordered_map<std::string, int> idx;
    for (int i = 0; i < (int)headers.size(); ++i) idx[headers[i]] = i;
    if (!idx.count("action_taken")) throw std::runtime_error("Missing action_taken column");

    std::unordered_set<std::string> drop = {
        "action_taken", "loan_amount", "income", "loan_term", "property_value"
    };

    std::vector<std::string> remain;
    for (const auto& h : headers) {
        if (!drop.count(h)) remain.push_back(h);
    }
    int start = 1;
    int end = std::min((int)remain.size(), 19);
    std::vector<std::string> feature_names;
    for (int i = start; i < end; ++i) feature_names.push_back(remain[i]);
    if (feature_names.empty()) throw std::runtime_error("No feature columns selected");

    ParsedDataset ds;
    ds.feature_names = feature_names;
    ds.cols = (int)feature_names.size();

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto cells = split_csv_line(line);
        if ((int)cells.size() != (int)headers.size()) continue;

        bool skip = false;
        for (const auto& name : feature_names) if (bad_cell(cells[idx[name]])) { skip = true; break; }
        if (skip || bad_cell(cells[idx["action_taken"]])) continue;

        try {
            for (const auto& name : feature_names) ds.X.push_back(std::stof(cells[idx[name]]));
            ds.y.push_back(std::stoi(cells[idx["action_taken"]]));
            ds.rows++;
        } catch (...) {
            continue;
        }
    }
    return ds;
}
