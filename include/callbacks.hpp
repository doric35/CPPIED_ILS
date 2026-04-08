#pragma once

#include "structures.hpp"
#include "cppied_instance.hpp"
#include "utils.hpp"

struct CPPIEDCallbacks{
    std::function<void(const cppied_solution& sol, const std::any& context)> onSatisfy;
    std::function<void(std::ofstream& stream, std::vector<std::string>& msg)> onLog;
    std::function<bool(const cppied_solution& sol, const std::any& context)> stopCriteria;
};