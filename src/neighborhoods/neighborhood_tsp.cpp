#include "../../include/neighborhoods/neighborhood_tsp.hpp"

bool neighborhood_tsp::local_search(cppied_solution &pSol) {
    split(pSol);

    int tsp_dimensions = static_cast<int>(pSol.path.size()) * 3 + 3;
    Eigen::MatrixXi C(tsp_dimensions, tsp_dimensions);
    C.setConstant(pSol.cost.length);
    path_to_tsp(pSol, C);
    assert(C.isApprox(C.transpose()));

    std::vector<int> tsp_solution(tsp_dimensions, 0);
    std::iota(tsp_solution.begin(), tsp_solution.end(), 1);

    tsp_solution = std::move(tspEngine.lkh(C,tsp_solution));
    cppied_solution trial = pSol;
    tsp_to_path(trial, tsp_solution);
    trial.cost = geometry.cost(trial);
    if (trial.cost < pSol.cost) {
        pSol = std::move(trial);
        return true;
    }
    return false;
}

void neighborhood_tsp::split(cppied_solution &pSol) {
    int max_split = int(std::sqrt(static_cast<double>(pSol.path.size())));
    std::vector<segment> new_sol;
    new_sol.reserve(pSol.path.size() + max_split);

    std::vector<int> selection(pSol.path.size());
    std::iota(selection.begin(), selection.end(), 0);

    auto cmp = [&](int i, int j){
        return geometry.cost(pSol.path[i]) > geometry.cost(pSol.path[j]);
    };
    std::partial_sort(selection.begin(),
                      std::next(selection.begin(), max_split),
                      selection.end(), cmp);

    std::vector<int> split_selection(selection.begin(),
                                     std::next(selection.begin(), max_split));
    std::sort(split_selection.begin(), split_selection.end());

    int i=0, j=0;
    for(;i<pSol.path.size();i++){
        if (j< split_selection.size() && i==split_selection[j]){
            if (geometry.cost(pSol.path[i]) <= cost_t{0,0}){
                new_sol.push_back(pSol.path[i]);
            } else if (geometry.cost(pSol.path[i]) == cost_t{1,0}){
                direction dir = geometry.get_direction(pSol.path[i]);
                segment s1 = {pSol.path[i].source, NULL_NODE};
                segment s2 = {pSol.path[i].target, NULL_NODE};
                geometry.correct_segment_direction(s1, dir);
                geometry.correct_segment_direction(s2,dir);
                new_sol.push_back(s1);
                new_sol.push_back(s2);
            } else if (geometry.cost(pSol.path[i]) == cost_t{2,0}){
                int mid = (pSol.path[i].target - pSol.path[i].source) / 2;
                mid += pSol.path[i].source;
                direction dir = geometry.get_direction(pSol.path[i]);
                segment s1 = {pSol.path[i].source, NULL_NODE};
                geometry.correct_segment_direction(s1, dir);
                segment s2 = {mid, pSol.path[i].target};
                new_sol.push_back(s1);
                new_sol.push_back(s2);
            } else {
                int first = std::min(pSol.path[i].source, pSol.path[i].target);
                int last = std::max(pSol.path[i].source, pSol.path[i].target);
                std::uniform_int_distribution<> d(first,last);
                int split_point = d(rng);

                segment s1{}, s2{};
                if (split_point == first) {
                    s1 = {first, NULL_NODE};
                    s2 = {first + 1, last};
                } else if (split_point == last){
                    s1 = {first, last-1};
                    s2 = {last, NULL_NODE};
                } else {
                    s1 = {first, split_point};
                    s2 = {split_point+1, last};
                    if (s2.source == last)
                        s2.target = NULL_NODE;
                }
                direction ref_dir = geometry.get_direction(pSol.path[i]);
                geometry.correct_segment_direction(s1, ref_dir);
                geometry.correct_segment_direction(s2, ref_dir);
                if (s1.source == pSol.path[i].source){
                    new_sol.push_back(s1);
                    new_sol.push_back(s2);
                } else {
                    new_sol.push_back(s2);
                    new_sol.push_back(s1);
                }
            }
            j++;
        } else
            new_sol.push_back(pSol.path[i]);
    }
    pSol.path.swap(new_sol);
}

void neighborhood_tsp::path_to_tsp(const cppied_solution &pSol,
                                   Eigen::Ref<Eigen::MatrixXi> C) {
    //Dummy nodes
    C(0,1) = 0;
    C(1,2) = 0;

    //Connect to initial position
    C(2, 3) = 0;

    //Connect all segment endpoints to 0 with null cost except for first segment which cant connect to end.
    for (int i=1; i<pSol.path.size(); i++){
        int mat_tsp_idx = (i+1) * 3;
        C(0, mat_tsp_idx) = 0;
        C(0, mat_tsp_idx+2) =0;
    }

    //Pair-wise segment endpoints connection
    for (int i =0; i< pSol.path.size(); i++){
        int tsp_idx = (i+1) * 3;
        C(tsp_idx, tsp_idx+1) = 0;
        C(tsp_idx+1, tsp_idx+2) = 0;

        segment i_reversed = geometry.flip_segment(pSol.path[i]);
        for (int j=i+1; j< pSol.path.size(); j++){
            int other_idx = (j+1) * 3;
            segment j_reversed = geometry.flip_segment(pSol.path[j]);
            if (i > 0){
                //Just for non start positions
                C(tsp_idx, other_idx) = geometry.dist(
                        i_reversed,
                        pSol.path[j]).length;
                C(tsp_idx, other_idx+2) = geometry.dist(
                        i_reversed,
                        j_reversed).length;
            }
            C(tsp_idx+2, other_idx) = geometry.dist(
                    pSol.path[i],
                    pSol.path[j]).length;
            C(tsp_idx+2, other_idx+2) = geometry.dist(
                    pSol.path[i],
                    j_reversed).length;
        }
    }
    //Set inferior matrix
    for (int i = 0; i < C.rows(); ++i) {
        for (int j = i + 1; j < C.cols(); ++j) {
            C(j, i) = C(i, j);
        }
    }
}

segment neighborhood_tsp::tour_id_to_segment(cppied_solution &pSol, int i) {
    assert(i > 3);

    if (i % 3 == 1)
        return pSol.path[(i-2) / 3];
    else if (i % 3 == 0)
        return geometry.flip_segment(pSol.path[(i-6) / 3]);
    else
        throw std::runtime_error("Encountered segment mid point in tsp solution.");
}

void neighborhood_tsp::tsp_to_path(cppied_solution &pSol,
                                   std::vector<int>& tour) {
    auto first_elem = std::find(tour.begin(), tour.end(), 1);
    if (first_elem != std::prev(tour.end())
        && *std::next(first_elem)!=2)
        std::reverse(tour.begin(), tour.end());
    first_elem = std::find(tour.begin(), tour.end(), 1);
    if (first_elem != tour.begin())
        std::rotate(tour.begin(), first_elem, tour.end());

    std::vector<segment> new_sol;
    new_sol.reserve(pSol.path.size());

    assert(tour[0] == 1 && tour[1] == 2 && tour[2] == 3);
    for (int i=3; i<tour.size(); i+=3){
        segment s = tour_id_to_segment(pSol, tour[i]);
        new_sol.push_back(s);
    }
    pSol.path.swap(new_sol);
}