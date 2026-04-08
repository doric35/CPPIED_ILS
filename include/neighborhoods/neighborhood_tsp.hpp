#pragma once

#include "neighborhood.hpp"
#include "../engines/tsp_engine.hpp"
#include "neighborhood_trim.hpp"
#include "neighborhood_cut.hpp"

class neighborhood_tsp : public neighborhood{
public:
    neighborhood_tsp(cppied_context& c, cppied_instance& i) :
        neighborhood(c,i), tspEngine(ctx){}

    bool local_search(cppied_solution&) override;
    void split(cppied_solution&);
    void path_to_tsp(const cppied_solution&,
                     Eigen::Ref<Eigen::MatrixXi> C);
    void tsp_to_path(cppied_solution&,
                     std::vector<int>& tour);
    segment tour_id_to_segment(cppied_solution&, int i);
protected:
    tsp_engine tspEngine;
};