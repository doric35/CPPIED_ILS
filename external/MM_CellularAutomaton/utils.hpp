#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <map>
#include <vector>
#include <random>
#include <cstdlib>
#include <utility>
#include <type_traits>
#include <functional>
#include <any>
#include <optional>

#define ALGORITHM_LIST \
    X(UNIFORM)         \
    X(CELLULAR_AUTOMATON)

enum class algorithm_type{
#define X(name) name,
    ALGORITHM_LIST
#undef X
};

inline std::string to_string(algorithm_type t){
    switch (t) {
#define X(name) case algorithm_type::name: return #name;
        ALGORITHM_LIST
#undef X
    }
    return "UNKNOWN";
}

inline algorithm_type to_type(const std::string& t){
#define X(name) if (t == #name) return algorithm_type::name;
    ALGORITHM_LIST
#undef X
    throw std::invalid_argument("Unkown algorithm type: " + t);
}



inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t");
    size_t end = s.find_last_not_of(" \t");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

inline void read_configuration(const std::string& filename,
                        std::map<std::string, std::string>& configuration){
    std::ifstream config_stream(filename);
    if (!config_stream.is_open()){
        std::cerr << "[ERROR]: Could not open the configuration file.\n";
        throw std::runtime_error("File error.\n");
    }

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
}