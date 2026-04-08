#include "../../include/engines/coverage_engine.hpp"

void coverage_engine::reset(cppied_solution &pSol) {
    pSol.coverage.setZero();

    for (auto& seg: pSol.path)
        insert(pSol, seg);
}

void coverage_engine::insert(cppied_solution& sol, const segment& seg) {
    auto func = [&](
            cppied_solution& pSol, int v){
        update_vertex_insert(pSol, v);
    };
    apply(sol, seg, func);
}

double coverage_engine::over_coverage(cppied_solution &sol, const segment &seg) {
    double oc_seg = 0.0;
    int set_cover_size = 0;
    auto func = [&](
            cppied_solution& pSol, int v){
        add_over_coverage_contribution(pSol, v, oc_seg, set_cover_size);
    };
    apply(sol, seg, func);
    if (set_cover_size <= 0)
        return 0.0;
    return oc_seg / set_cover_size;
}

void coverage_engine::add_over_coverage_contribution(cppied_solution &pSol,
                                                     int v,
                                                     double &acc_oc,
                                                     int &acc_size) {
    for (SMdIt c(problem.s_pod, v); c; ++c) {
        acc_oc += std::max(0.0,
                           pSol.coverage(c.index()) - problem.req(c.index()));
        ++acc_size;
    }
}

void coverage_engine::update_vertex_insert(cppied_solution &pSol, int v) {
    for (SMdIt c(problem.s_pod, v); c; ++c)
        pSol.coverage(c.index()) += c.value();
}

void coverage_engine::remove(cppied_solution &sol,const segment &seg) {
    auto func = [&](
            cppied_solution& pSol, int v){
        update_vertex_remove(pSol, v);
    };
    apply(sol, seg, func);
}

void coverage_engine::update_vertex_remove(cppied_solution &pSol, int v) {
    for (SMdIt c(problem.s_pod, v); c; ++c)
        pSol.coverage(c.index()) -= c.value();
}

bool coverage_engine::can_remove(cppied_solution &sol, const segment &s) const{
    auto func = [&](
            cppied_solution& pSol, int v){
        return can_remove_vertex(pSol, v);
    };
    return verify(sol, s, func);
}

bool coverage_engine::can_remove_vertex(const cppied_solution & pSol, int v) const{
    bool sat = true;
    for (SMdIt c(problem.s_pod, v); c; ++c)
        sat = sat && (pSol.coverage(c.index()) - c.value() >= problem.req(c.index()));
    return sat;
}

bool coverage_engine::insertion_satisfies(cppied_solution &sol, const segment &seg, const std::vector<int> &set_cover) {

    insert(sol,seg);
    bool sat = true;

    for (int c : set_cover)
        sat &= sol.coverage(c) >= problem.req(c);


    remove(sol,seg);
    return sat;
}

void coverage_engine::unsatisfied_cells(const cppied_solution &sol, std::vector<int> &out) const {
    out.clear();

    for (int i = 0; i < sol.coverage.size(); i++)
        if (sol.coverage(i) < problem.req(i))
            out.push_back(i);
}

void coverage_engine::filter_unsatisfied_cells(cppied_solution &sol,
                                               std::vector<int> &unsat) {
    std::vector<int> filtered_unsat;
    filtered_unsat.reserve(unsat.size());
    auto filter = [&](int c){
        return sol.coverage(c) >= problem.req(c);
    };
    std::erase_if(unsat, filter);
}

void coverage_engine::unsatisfied_cells(cppied_solution &sol,
                                        const segment &ref,
                                        std::vector<int> &out) const {
    auto func = [&](
            cppied_solution& pSol, int v){
        push_vertex_unsatisfied_cells(pSol, v, out);
    };
    apply(sol, ref, func);
}

void coverage_engine::push_vertex_unsatisfied_cells(const cppied_solution &pSol, int v, std::vector<int> &out) const {
    for (SMdIt c(problem.s_pod, v); c; ++c)
        if (pSol.coverage(c.index()) < problem.req(c.index()))
            out.push_back(c.index());
}

bool coverage_engine::satisfy(const cppied_solution &sol) const {
    return (sol.coverage.array() >= problem.req.array()).all();
}

void coverage_engine::unsatisfied_rectangle(const std::vector<int> &unsat, iRectangle &box) {
    for (int c: unsat){
        box.ul.x = std::min(static_cast<int>(problem.cells(0,c)),
                                box.ul.x);
        box.ul.y = std::min(static_cast<int>(problem.cells(1,c)),
                                box.ul.y);
        box.lr.x = std::max(static_cast<int>(problem.cells(0,c)+1),
                                box.lr.x);
        box.lr.y = std::max(static_cast<int>(problem.cells(1,c)+1),
                                box.lr.y);
    }
}

bool coverage_engine::intersect_mask(cppied_solution& sol,
                                     const segment &s,
                                     Eigen::Ref<Eigen::VectorXi> mask) {
    auto func = [&](
            cppied_solution& pSol, int v){
        bool intersection_free = true;
        for (SMdIt c(problem.s_pod, v); c; ++c)
            intersection_free = intersection_free && mask(c.index()) == 0;

        return intersection_free;
    };
    return !verify(sol, s, func);
}

void coverage_engine::mask(cppied_solution& sol,
                           const segment &s,
                           Eigen::Ref<Eigen::VectorXi> mask) {
    auto func = [&](
            cppied_solution& pSol, int v){
        for (SMdIt c(problem.s_pod, v); c; ++c)
            mask(c.index()) = 1;
    };
    apply(sol, s, func);
}

//void coverage_engine::unsat_boundaries(cppied_solution &pSol, segment& pSeg, iRectangle pBounds) {
//    if (is_node(pSeg.target)){
//        int first = std::min(pSeg.source, pSeg.target);
//        int last = std::max(pSeg.source, pSeg.target);
//        for (int v = first; v<=last; v++)
//            unsat_boundaries(pSol, v, pBounds);
//    } else
//        unsat_boundaries(pSol, pSeg.source, pBounds);
//}
//
//void coverage_engine::unsat_boundaries(cppied_solution &pSol, int v, iRectangle pBounds) {
//    for (SMdIt c(problem.s_pod, v); c; ++c)
//        if (pSol.coverage(c.index()) < problem.req(c.index())) {
//            pBounds.ul.x = std::min(static_cast<int>(problem.cells(0,c.index())),
//                                    pBounds.ul.x);
//            pBounds.ul.y = std::min(static_cast<int>(problem.cells(1,c.index())),
//                                    pBounds.ul.y);
//            pBounds.lr.x = std::max(static_cast<int>(problem.cells(0,c.index())),
//                                    pBounds.lr.x);
//            pBounds.lr.y = std::min(static_cast<int>(problem.cells(1,c.index())),
//                                    pBounds.lr.y);
//        }
//}