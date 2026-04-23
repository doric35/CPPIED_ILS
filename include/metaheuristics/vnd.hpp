#pragma once

#include "../cppied_method_base.hpp"
#include "../neighborhoods/neighborhood.hpp"
#include "../neighborhoods/neighborhood_trim.hpp"
#include "../neighborhoods/neighborhood_cut.hpp"
#include "../neighborhoods/neighborhood_reduce.hpp"
#include "../neighborhoods/neighborhood_n1.hpp"
#include "../neighborhoods/neighborhood_nested.hpp"
#include "../neighborhoods/neighborhood_n12.hpp"
#include "../neighborhoods/neighborhood_n21.hpp"
#include "../neighborhoods/neighborhood_tsp.hpp"
#include "../neighborhoods/neighborhood_gtsp.hpp"
#include "../neighborhoods/neighborhood_r.hpp"

class vnd : public cppied_method_base{
public:
    using cppied_method_base::cppied_method_base;

    void initialize() override {
        std::string configuration = ctx.config["ALGORITHM_CONFIG"];
        assert(configuration.size() == 12);
        N_setup.push_back(std::make_unique<neighborhood_reduce>(ctx, problem));
        N_setup.push_back(std::make_unique<neighborhood_trim>(ctx, problem));
        N_setup.push_back(std::make_unique<neighborhood_cut>(ctx, problem));
        if (configuration[1] == '1')
            N_simple.push_back(std::make_unique<neighborhood_n1>(ctx, problem));
        if (configuration[2] == '1')
            N_simple.push_back(std::make_unique<neighborhood_n12>(ctx, problem));
        if (configuration[3] == '1')
            N_simple.push_back(std::make_unique<neighborhood_n21>(ctx, problem));
        if (configuration[4] == '1')
            N_nested.push_back(std::make_unique<neighborhood_nested>(ctx, problem));
        if (configuration[5] == '1')
            N_large.push_back(std::make_unique<neighborhood_tsp>(ctx, problem));
        if (configuration[6] == '1')
            N_large.push_back(std::make_unique<neighborhood_gtsp>(ctx, problem));
        if (configuration[7] == '1')
            N_large.push_back(std::make_unique<neighborhood_r>(ctx, problem));
    }

    void terminate() override {}

    void d_solve(cppied_solution& pSolution) override{
        throw std::runtime_error("Class no implemented for a direct d_solve call.");
    }
    bool search(cppied_solution& pSolution);
    size_t neighborhoods_count(){return N_setup.size() + N_simple.size() + N_nested.size() + N_large.size();}

    void find_local_optima(cppied_solution& pSol, const std::unique_ptr<neighborhood>& N);

protected:
    std::vector<std::unique_ptr<neighborhood>> N_setup{};
    std::vector<std::unique_ptr<neighborhood>> N_simple{};
    std::vector<std::unique_ptr<neighborhood>> N_nested{};
    std::vector<std::unique_ptr<neighborhood>> N_large{};
};