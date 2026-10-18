#pragma once

#include "../cppied_method_base.hpp"
#include "HeuristicFactory.hpp"

class vnd : public cppied_method_base{
public:
    vnd(cppied_context& c, cppied_instance& i) :
     cppied_method_base(c, i), factory(std::bitset<7>(reverse_string(ctx.config["ALGORITHM_CONFIG"].substr(1, 7)))){}

    void initialize() override {}
    void terminate() override {}
    void d_solve(cppied_solution& pSolution) override{
        throw std::runtime_error("Class no implemented for a direct d_solve call.");
    }
    bool search(cppied_solution& pSolution);
    [[nodiscard]] size_t neighborhoods_count() const{ return factory.size(); }

    void find_local_optima(cppied_solution& pSol, const std::unique_ptr<neighborhood>& N);

protected:
    LocalSearchFactory factory;
private:
    void simpleNeighborhoodsDescent(cppied_solution &pSol);
    void nestedNeighborhoodsDescent(cppied_solution &pSol);
    void largeNeighborhoodsDescent(cppied_solution &pSol);
};