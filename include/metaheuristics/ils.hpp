#pragma once

#include "../cppied_method_base.hpp"
#include "../heuristics/luby.hpp"
#include "vnd.hpp"
#include "../perturbations/perturbation.hpp"
#include "../perturbations/perturbation_ri.hpp"
#include "../perturbations/perturbation_pt.hpp"
#include "../perturbations/perturbation_db.hpp"
#include "../restart/restart_backtrack.hpp"
#include "../construction/dp_sweeper.hpp"

class ils : public cppied_method_base{
public:
    ils(cppied_context& c, cppied_instance& i) :
        cppied_method_base(c, i), ls(c, i), L(), incumbent(){}
    void initialize() override {
        std::string configuration = ctx.config["ALGORITHM_CONFIG"];
        assert(configuration.size() == 12);
        ls.initialize();
        if (configuration[8] == '1')
            P.push_back(std::make_unique<perturbation_ri>(ctx, problem));
        if (configuration[9] == '1')
            P.push_back(std::make_unique<perturbation_pt>(ctx, problem));
        if (configuration[10] == '1')
            P.push_back(std::make_unique<perturbation_db>(ctx, problem));
        if (configuration[11] == '1')
            R.push_back(std::make_unique<restart_backtrack>(ctx, problem));
    }
    void terminate() override {

    }
    void d_solve(cppied_solution& pSolution) override;
    bool local_search(cppied_solution& pSol);
    bool stopping_criterion();

    template<class T>
    T* uniform_sample(const std::vector<std::unique_ptr<T>>& candidates){
        std::uniform_int_distribution<> d(0, int(candidates.size())-1);
        int k = d(rng);
        return candidates[k].get();
    };
    
protected:
    vnd ls;
    std::vector<std::unique_ptr<perturbation>> P{};
    std::vector<std::unique_ptr<restarts>>     R{};
    luby L;
    int iterations = 1;
    cppied_solution incumbent;

private:
    void constructInitialSolution();
    void rootLocalSearch();
    void iteratedSearch(cppied_solution &pSolution);
    cppied_solution perturbate(const cppied_solution &pSolution);
    cppied_solution restart();

    void setLocalSearchCallbacks();
    int setIncumbent(const cppied_solution& pSolution);
    void setTerminalIncumbent();
    void setInitialIncumbent();

    int getElapsedTime();

    bool isConstructOnly(){
        return R.empty() && P.empty() && ls.neighborhoods_count() <=3;
    }
    bool isLocalSearchOnly() {
        return R.empty() && P.empty();
    }
};