#pragma once

#include "utils.hpp"
#include "cppied_method_base.hpp"

class SolverFactory {
public:
    using Creator = std::function<std::unique_ptr<cppied_method_base>(cppied_context&, cppied_instance&)>;

    static SolverFactory& instance() {
        static SolverFactory factory;
        return factory;
    }

    void register_solver(const std::string& name, Creator creator) {
        if (registry_.count(name))
            throw std::runtime_error("Solver already registered: " + name);

        registry_[name] = std::move(creator);
    }

    std::unique_ptr<cppied_method_base> create(
            const std::string& name,
            cppied_context& ctx,
            cppied_instance& problem) const
    {
        auto it = registry_.find(name);
        if (it == registry_.end())
            throw std::runtime_error("Unknown solver: " + name);

        return it->second(ctx, problem);
    }

    void debug_print() const {
        for (const auto& [k, _] : registry_) {
            std::cout << k << std::endl;
        }
    }

private:
    std::unordered_map<std::string, Creator> registry_;
};

#define REGISTER_SOLVER(NAME, CLASS) \
    namespace { \
        const bool registered_##CLASS = [](){ \
            SolverFactory::instance().register_solver(NAME, \
                [](cppied_context& ctx, cppied_instance& prob){ \
                    return std::make_unique<CLASS>(ctx, prob); \
                }); \
            return true; \
        }(); \
    }