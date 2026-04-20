#pragma once

#include "structures.hpp"
#include "cppied_instance.hpp"
#include "engines/coverage_engine.hpp"
#include "engines/path_engine.hpp"
#include "callbacks.hpp"

class cppied_method_base {
protected:
    //Data
    cppied_context  &ctx;
    cppied_instance &problem;

    //Engines
    coverage_engine coverage;
    path_engine     geometry;

    std::random_device rd;
    std::mt19937       rng;

    CPPIEDCallbacks callbacks;
    algorithm_flag  status;
public:
    cppied_method_base(cppied_context& c, cppied_instance& i) :
        ctx (c), problem(i), coverage(i), geometry(i), rd(), rng(rd()),
        callbacks({}), status(algorithm_flag::NONE){}

    virtual ~cppied_method_base() = default;

    virtual void initialize() = 0;
    virtual void terminate() = 0;
    virtual void d_solve(cppied_solution& pSolution) = 0;

    void solve(cppied_solution& solution);
    void validate_solution(cppied_solution& pSolution);

    void set_callbacks(CPPIEDCallbacks& c){
        callbacks = c;
    }

    int remaining_time();
};
