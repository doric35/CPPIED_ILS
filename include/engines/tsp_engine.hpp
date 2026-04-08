#pragma once

#include "../structures.hpp"

namespace tsp{
    struct edge{
        int source;
        int target;
        int cost;
    };

    struct node{
        int id;
        int cluster_id;
        segment s;
    };

    struct cluster{
        int id;
        std::vector<int> nodes;
    };

    struct GTSPGraph{
        std::vector<node> nodes;
        std::vector<cluster> clusters;

        Eigen::MatrixXi cost;
        int penalty;
    };
}

class tsp_engine{
public:
    tsp_engine(const cppied_context& c) : ctx(c){}
    std::vector<int> lkh(Eigen::Ref<Eigen::MatrixXi> C,
                         std::vector<int>& warm_start);
    void gtsp_to_atsp(tsp::GTSPGraph& graph);
    void gtsp_to_sgtsp(tsp::GTSPGraph& graph);
    void sgtsp_to_atsp(tsp::GTSPGraph& graph);

    void tsp_tour_to_gtsp_tour(std::vector<int>& tour,
                               std::vector<std::shared_ptr<tsp::cluster>>& map);
    void write_configuration(std::vector<int>& warm_start);
    void write_instance_file(Eigen::Ref<Eigen::MatrixXi> C);
    void write_data(Eigen::Ref<Eigen::MatrixXi> C,
                         std::ofstream& problem_stream);
    void write_initial_tour(std::vector<int>& warm_start);
    void read_solution(std::vector<int>& container);
private:
    const cppied_context& ctx;
};