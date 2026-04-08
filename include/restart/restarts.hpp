#pragma once

#include "../cppied_method_base.hpp"

class restarts : public cppied_method_base{
public:
    using cppied_method_base::cppied_method_base;
    void initialize() override {
        throw std::runtime_error("Not implemented for neighborhoods interface.");
    }
    void terminate() override {
        throw std::runtime_error("Not implemented for neighborhoods interface.");
    }
    void d_solve(cppied_solution& pSolution) override {
        throw std::runtime_error("Not implemented for neighborhoods interface.");
    }
    virtual ~restarts() = default;
    virtual void restart(cppied_solution& pSolution) = 0;
protected:

};