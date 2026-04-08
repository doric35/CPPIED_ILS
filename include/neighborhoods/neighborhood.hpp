#pragma once

#include "../cppied_method_base.hpp"
#include "../heuristics/ris_heuristic.hpp"

class neighborhood : public cppied_method_base{
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
    virtual ~neighborhood() = default;
    virtual bool local_search(cppied_solution& pSolution) = 0;

    using sVecIt = std::vector<segment>::const_iterator;
    segment trim(cppied_solution&, sVecIt seg);
    segment trim(cppied_solution&, segment seg);

    template<class F>
    void for_each_segment(const iRectangle& box,
                          const segment& ref,
                          F&& f){
        int row = std::max(0, box.ul.y - (problem.max_range-1));
        int bound = std::min(geometry.n_rows, box.lr.y + (problem.max_range-1));
        int no_good;
        if (is_node(ref.source))
            no_good = geometry.is_horizontal(ref) ? ref.source / geometry.n_cols : -1;
        else
            no_good = -1;

        //Horizontal segments
        for (; row <= bound; ++row){
            if (row != no_good){
                segment s{
                        row * geometry.n_cols + box.ul.x,
                        row * geometry.n_cols + box.lr.x - 1
                };

                if (s.source == s.target)
                    s.target = NULL_NODE;

                f(s);
            }
        }

        //Vertical segments
        int col = std::max(0, box.ul.x - (problem.max_range - 1));
        bound = std::min(geometry.n_cols, box.lr.x + (problem.max_range-1));
        if (is_node(ref.source))
            no_good = geometry.is_horizontal(ref) ? -1 :
                      (ref.source - geometry.horizontal_bound) / geometry.n_rows;
        else
            no_good = -1;
        for (; col <= bound; ++col){
            if (col != no_good){
                segment s{
                        geometry.horizontal_bound + (col * geometry.n_rows) + box.ul.y,
                        geometry.horizontal_bound + col * geometry.n_rows + box.lr.y - 1
                };

                if (s.source == s.target)
                    s.target = NULL_NODE;

                f(s);
            }
        }
    }
};
