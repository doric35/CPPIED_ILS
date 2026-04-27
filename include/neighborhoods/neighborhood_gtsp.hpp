#pragma once

#include "neighborhood.hpp"
#include "../engines/tsp_engine.hpp"
#include "neighborhood_trim.hpp"
#include "neighborhood_cut.hpp"

class neighborhood_gtsp: public neighborhood{
public:
    neighborhood_gtsp(cppied_context& c, cppied_instance& i) :
        neighborhood(c,i), tspEngine(ctx){}

    bool local_search(cppied_solution&) override;
    void cluster_selection(cppied_solution&,
                           std::vector<int>& selection,
                           std::vector<std::vector<segment>>& replacements);
    void split(cppied_solution &pSol);
    void build_clusters(const cppied_solution& pSol,
                        tsp::GTSPGraph& target,
                        std::vector<std::vector<segment>>& replacements,
                        std::vector<int>& selection,
                        std::vector<int>& warm_start);
    void add_replacements(cppied_solution &,std::vector<segment>& candidates);
    void set_gtsp_costs(tsp::GTSPGraph& graph);
    void tsp_to_path(cppied_solution&,
                     std::vector<int>& tour,
                     tsp::GTSPGraph& graph);
    std::array<segment, 3> segment_to_nodes(const segment& s);
    segment nodes_to_segment(const std::vector<segment>& nodes);
    cost_t gain(const cppied_solution& pSol, sVecIt s, int M);
protected:
    tsp_engine tspEngine;
};