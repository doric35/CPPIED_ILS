#pragma once

#include "../cppied_method_base.hpp"

class Restart : public cppied_method_base{
private:
    cppied_solution& solution;
public:
    Restart(cppied_context& c, cppied_instance& i, cppied_solution& s) :
            cppied_method_base(c, i), solution(s){}
    void initialize() override {
        throw std::runtime_error("Not implemented for neighborhoods interface.");
    }
    void terminate() override {
        throw std::runtime_error("Not implemented for neighborhoods interface.");
    }
    void d_solve(cppied_solution& pSolution) override {
        throw std::runtime_error("Not implemented for neighborhoods interface.");
    }
    ~Restart() override = default;
    virtual void restart(cppied_solution& pSolution) = 0;
protected:

};