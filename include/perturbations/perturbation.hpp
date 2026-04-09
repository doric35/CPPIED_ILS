#pragma once

#include "../cppied_method_base.hpp"

class perturbation : public cppied_method_base {
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
    virtual ~perturbation() = default;

    void perturbate(cppied_solution&);

    virtual void d_perturbate(cppied_solution&) = 0;

    template<class F>
    void for_each_segment(const iRectangle& box,
                          F&& f){
        int row = std::max(0, box.ul.y - (problem.max_range-1));
        int bound = std::min(geometry.n_rows, box.lr.y + (problem.max_range-1));

        //Horizontal segments
        for (; row <= bound; ++row){
            segment s{
                    row * geometry.n_cols + box.ul.x,
                    row * geometry.n_cols + box.lr.x - 1
            };

            if (s.source == s.target)
                s.target = path_engine::NULL_NODE;

            f(s);
        }

        //Vertical segments
        int col = std::max(0, box.ul.x - (problem.max_range - 1));
        bound = std::min(geometry.n_cols, box.lr.x + (problem.max_range-1));
        for (; col <= bound; ++col){
            segment s{
                    geometry.horizontal_bound + col * geometry.n_rows + box.ul.y,
                    geometry.horizontal_bound + col * geometry.n_rows + box.lr.y - 1
            };

            if (s.source == s.target)
                s.target = path_engine::NULL_NODE;

            f(s);
        }
    }
};