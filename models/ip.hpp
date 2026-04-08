#include "gurobi_c++.h"
#include "../abstract_classes/cppied_method.hpp"

#ifndef CPPIED_IP_HPP
#define CPPIED_IP_HPP


class ip: protected cppied_method{
protected:
public:
    template<typename... Args>
    ip(Args&&... args) : cppied_method(std::forward<Args>(args)...){}

    void initialize() override{
        ;
    }

    void terminate() override{
        ;
    }

    void d_solve(cppied_solution& pSolution) override{
        ;
    }

};


#endif //CPPIED_IP_HPP
