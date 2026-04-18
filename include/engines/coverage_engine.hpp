#pragma once

#include "../structures.hpp"
#include "path_engine.hpp"
#include "../cppied_instance.hpp"

class coverage_engine{
public:
    coverage_engine(const cppied_instance& instance) : problem(instance){}
    void reset(cppied_solution& sol);
    void insert(cppied_solution& sol, const segment& seg);
    void remove(cppied_solution& sol, const segment& seg);
    double over_coverage(cppied_solution& sol, const segment& seg);

    template<class F>
    void apply(cppied_solution& sol, const segment& seg, F&& f) const{
        if (path_engine::is_node(seg.target)){
            int first = std::min(seg.source, seg.target);
            int last = std::max(seg.source, seg.target);
            for (int v = first; v<=last; v++)
                f(sol, v);
        } else {
            f(sol, seg.source);
        }
    }

    template<class F>
    bool verify(cppied_solution& sol, const segment& seg, F&& f) const{
        if (path_engine::is_node(seg.target)){
            int first = std::min(seg.source, seg.target);
            int last = std::max(seg.source, seg.target);
            for (int v = first; v<=last; v++)
                if (!f(sol,v))
                    return false;
            return true;
        } else {
            return f(sol, seg.source);
        }
    }

    bool can_remove(cppied_solution& sol, const segment& seg) const;
    bool insertion_satisfies(
            cppied_solution& sol,
            const segment& seg,
            const std::vector<int>& set_cover);
    void unsatisfied_cells(
            const cppied_solution& sol,
            std::vector<int>& out) const;
    void unsatisfied_cells(
            cppied_solution& sol,
            const segment& ref,
            std::vector<int>& out) const;
    void filter_unsatisfied_cells(
            cppied_solution& sol,
            std::vector<int>& unsat
            );
    void unsatisfied_rectangle(const std::vector<int>& unsat, iRectangle& box);
    bool intersect_mask(cppied_solution& pSol, const segment& s, Eigen::Ref<Eigen::VectorXi> mask);
    void mask(cppied_solution &pSol, const segment& s, Eigen::Ref<Eigen::VectorXi> mask);
    bool satisfy(const cppied_solution& sol) const;

protected:
    const cppied_instance& problem;
    
    void update_vertex_insert(cppied_solution&, int v);
    void update_vertex_remove(cppied_solution&, int v);
    void add_over_coverage_contribution(cppied_solution&, int v, double& acc_oc, int& acc_size);
    bool can_remove_vertex(const cppied_solution&, int v) const;
    void push_vertex_unsatisfied_cells(const cppied_solution& pSol, int v, std::vector<int>& out) const;
};
