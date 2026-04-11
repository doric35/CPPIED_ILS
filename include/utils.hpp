#pragma once

#include <iostream>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <list>
#include <set>
#include <tuple>
#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cmath>
#include <random>
#include <chrono>
#include <cstdlib>
#include <utility>
#include <type_traits>
#include <functional>
#include <any>
#include <opencv2/opencv.hpp>

using SMdIt = Eigen::SparseMatrix<double>::InnerIterator;
using SMiIt = Eigen::SparseMatrix<int>::InnerIterator;

template<typename T>
using MatrixXdRow = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class algo_exception : public std::exception {
private:
    std::string message_;

public:
    explicit algo_exception(const std::string& message)
            : message_(message) {}

    [[nodiscard]] const char* what() const noexcept override {
        return message_.c_str();
    }
};

class algo_flag : public std::exception {
private:
    std::string message_;
public:
    explicit algo_flag(const std::string& message)
            : message_(message) {}

    [[nodiscard]] const char* what() const noexcept override {
        return message_.c_str();
    }
};

template<typename T>
MatrixXdRow<T> read_matrix(const std::string& filename){
    std::ifstream instance_stream(filename);
    if (!instance_stream.is_open())
        throw std::runtime_error("Unable to open instance file.");

    int rows, cols;
    if (!(instance_stream >> rows >> cols))
        throw std::runtime_error("Failed to read matrix dimensions.");

    if (rows <= 0 || cols <= 0)
        throw std::runtime_error("Matrix dimensions must be positive.");

    MatrixXdRow<T> M(rows, cols);
    for (int i=0; i< rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (!(instance_stream >> M(i, j)))
                throw std::runtime_error("Not enough values while reading matrix at position (" +
                                         std::to_string(i) + "," + std::to_string(j) + ")");
        }
    }

    std::string extra;
    instance_stream >> extra;
    if (!extra.empty() && !(extra == "EOF" || extra == "eof"))
        throw std::runtime_error("File contains extra data beyond matrix size");

    instance_stream.close();
    return M;
}

inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t");
    size_t end = s.find_last_not_of(" \t");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

inline std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, delimiter)) {
        tokens.push_back(item);
    }

    return tokens;
}

inline std::string trim_space(std::string& s){
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
    return s;
}

enum class VerboseFlag {
    SUMMARY,
    ITERATIONS,
    SOLUTION,
    DEBUG,
    VIZ
};

const std::unordered_map<std::string, VerboseFlag> verbose_map = {
        {"SUMMARY", VerboseFlag::SUMMARY},
        {"ITERATIONS", VerboseFlag::ITERATIONS},
        {"SOLUTION", VerboseFlag::SOLUTION},
        {"DEBUG", VerboseFlag::DEBUG},
        {"VISUALIZE", VerboseFlag::VIZ}
};

inline std::unordered_map<std::string,std::string> read_configuration(const std::string& filename){
    std::unordered_map<std::string, std::string> configuration;

    std::ifstream config_stream(filename);
    if (!config_stream.is_open())
        throw std::runtime_error("Unable to open configuration file.");

    std::string line;
    int line_number = 0;
    while (std::getline(config_stream, line)){
        line_number++;
        line = trim(line);

        if (line.empty() || line[0] == '#') continue;

        auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos)
            throw std::runtime_error("Invalid configuration line " + std::to_string(line_number) + ": missing '='.");

        std::string key = trim(line.substr(0, eq_pos));
        std::string value = trim(line.substr(eq_pos+1));

        if (key.empty() || value.empty())
            throw std::runtime_error("Invalid configuration line " + std::to_string(line_number) + ": missing key or value.");

        configuration[key] = value;
    }

    config_stream.close();
    return configuration;
}

inline std::set<VerboseFlag> parse_verbose(const std::string& value) {
    std::set<VerboseFlag> result;

    auto tokens = split(value, ',');

    for (auto& token : tokens) {
        token = trim(token);
        token = trim_space(token);

        auto it = verbose_map.find(token);
        if (it != verbose_map.end()) {
            result.insert(it->second);
        } else {
            throw std::runtime_error("Unknown VERBOSE flag: " + token);
        }
    }

    return result;
}

inline void write_csv(const std::string& filepath, const std::vector<std::string>& log){
    std::string sep = ",";
    std::ofstream csv_stream(filepath, std::ios::app);

    if (!csv_stream.is_open())
        throw std::runtime_error("Failed to open file: " + filepath);

    for (int i=0; i< log.size()-1; i++)
        csv_stream << log[i] << sep;

    if (!log.empty())
        csv_stream << log.back() << std::endl;

    csv_stream.close();
}

template<typename T>
inline void write_solution(const std::string& filepath,
                           const std::map<std::string, std::string>& header,
                           T& solution){
    std::ofstream sol_stream(filepath, std::ios::out);
    if (!sol_stream.is_open())
        throw std::runtime_error("Failed to open file: " + filepath);

    for (auto& [k,v] : header)
        sol_stream << k << " : " << v << "\n";

    sol_stream << "\n";
    sol_stream << solution;
    sol_stream.close();
}

class timeout_error : public std::runtime_error {
public:
    explicit timeout_error(const std::string& message)
            : std::runtime_error(message) {}
};




