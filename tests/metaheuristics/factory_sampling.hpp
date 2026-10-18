#pragma once

#include <set>
#include <typeindex>
#include "../../include/metaheuristics/HeuristicFactory.hpp"

// HeuristicFactory pools only expose uniform sampling: the distinct classes a
// pool holds are recovered by sampling it repeatedly and keeping one instance
// per dynamic type. With 200 draws, missing one of k <= 3 classes has
// probability at most 3 * (2/3)^200.
template<typename T, typename Sampler>
std::vector<std::unique_ptr<T>> distinct_samples(Sampler&& sample,
                                                 cppied_context& c,
                                                 cppied_instance& i,
                                                 cppied_solution& s,
                                                 int draws = 200) {
    std::vector<std::unique_ptr<T>> distinct;
    std::set<std::type_index> seen;
    for (int k = 0; k < draws; ++k) {
        std::unique_ptr<T> candidate = sample()(c, i, s);
        const T& ref = *candidate;
        if (seen.insert(std::type_index(typeid(ref))).second)
            distinct.push_back(std::move(candidate));
    }
    return distinct;
}

// One instance of every perturbation class held by the factory, bound to s.
inline std::vector<std::unique_ptr<perturbation>> sample_perturbations(HeuristicFactory& factory,
                                                                       cppied_context& c,
                                                                       cppied_instance& i,
                                                                       cppied_solution& s) {
    if (factory.isWithoutPerturbation()) return {};
    return distinct_samples<perturbation>(
            [&factory] { return factory.samplePerturbationMethod(); }, c, i, s);
}

// One instance of every restart class held by the factory, bound to s.
inline std::vector<std::unique_ptr<Restart>> sample_restarts(HeuristicFactory& factory,
                                                             cppied_context& c,
                                                             cppied_instance& i,
                                                             cppied_solution& s) {
    if (factory.isWithoutRestart()) return {};
    return distinct_samples<Restart>(
            [&factory] { return factory.sampleRestartMethod(); }, c, i, s);
}
