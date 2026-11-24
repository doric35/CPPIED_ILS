#include "../../include/perturbations/perturbation_pt.hpp"

void perturbation_pt::d_perturbate() {
    std::vector<int> sampling = sampleCandidates();
    replaceCandidates(sampling);

    solution.cost = geometry.cost(solution);
    neighborhood_trim N(ctx, problem, solution);

    bool improved = true;
    while (improved)
        improved = N.local_search();
}

std::vector<int> perturbation_pt::sampleCandidates(){
    auto indices = std::views::iota(1, static_cast<int>(solution.path.size()));
    std::vector<int> candidates(indices.begin(), indices.end());
    std::shuffle(candidates.begin(), candidates.end(), rng);
    std::vector<int> sampling(candidates.begin(), candidates.begin() + static_cast<int>(K));
    std::sort(sampling.begin(), sampling.end());
    return sampling;
}

void perturbation_pt::replaceCandidates(const std::vector<int>& candidates){
    std::vector<segment> path;
    int j=0;
    for (int i=0; i< candidates.size(); ++i){
        path.insert(path.end(), solution.path.begin() + j, solution.path.begin() + candidates[i]);
        replaceSegment(path, candidates[i]);
        j = candidates[i] + 1;
    }
    path.insert(path.end(), solution.path.begin() + j, solution.path.end());
    solution.path.swap(path);
}

void perturbation_pt::replaceSegment(std::vector<segment> &new_sol,
                                      int i) {
    coverage.remove(solution, solution.path[i]);
    std::vector<segment> replacements = getReplacements(solution.path[i]);
    if (!replacements.empty()) {
        setReplacementsCoverage(replacements);
        direction dir = getReplacementDirection(new_sol, replacements);
        pushBackParallelTracks(new_sol, replacements, dir);
    }
}

std::vector<segment> perturbation_pt::getReplacements(const segment &s) {
    std::vector<int> unsatisfied_cells;
    std::vector<segment> replacements;
    coverage.unsatisfied_cells(solution, s, unsatisfied_cells);
    iRectangle box;
    coverage.unsatisfied_rectangle(unsatisfied_cells, box);
    if (geometry.is_horizontal(s))
        replacements = getVerticalCoverReplacements(box);
    else
        replacements = getHorizontalCoverReplacements(box);
    return replacements;
}

std::vector<segment> perturbation_pt::getHorizontalCoverReplacements(const iRectangle &box) {
    std::vector<segment> replacements;
    int row = std::max(0, box.ul.y - (problem.max_range-1));
    int bound = std::min(geometry.n_rows, box.lr.y + (problem.max_range-1));

    //Horizontal segments
    for (; row <= bound; ++row){
        segment s{row * geometry.n_cols + box.ul.x,
                row * geometry.n_cols + box.lr.x - 1 };
        s.target = s.source == s.target ? path_engine::NULL_NODE : s.target;
        replacements.push_back(s);
    }
    return replacements;
}

std::vector<segment> perturbation_pt::getVerticalCoverReplacements(const iRectangle &box) {
    std::vector<segment> replacements;
    int col = std::max(0, box.ul.x - (problem.max_range - 1));
    int bound = std::min(geometry.n_cols, box.lr.x + (problem.max_range-1));
    for (; col <= bound; ++col){
        segment s{geometry.horizontal_bound + col * geometry.n_rows + box.ul.y,
                geometry.horizontal_bound + col * geometry.n_rows + box.lr.y - 1};

        s.target = s.source == s.target ? path_engine::NULL_NODE : s.target;
        replacements.push_back(s);
    }
    return replacements;
}

void perturbation_pt::setReplacementsCoverage(const std::vector<segment> &replacementSet) {
    for (auto s : replacementSet)
        coverage.insert(solution, s);
}

direction perturbation_pt::getReplacementDirection(const std::vector<segment> &new_sol,
                                                   const std::vector<segment> &replacementSet) {
    direction dir = geometry.get_direction(replacementSet.front());
    cost_t insertion_cost = geometry.dist(new_sol.back(), replacementSet.front());
    cost_t reversed_cost = geometry.dist(new_sol.back(), geometry.flip_segment(replacementSet.front()));
    if (insertion_cost == reversed_cost){
        std::uniform_int_distribution<> d(0,1);
        int flip = d(rng);
        if (flip)
            dir = flip_direction(dir);
    } else if (reversed_cost < insertion_cost)
        dir = geometry.get_direction(geometry.flip_segment(replacementSet.front()));

    return dir;
}

void perturbation_pt::pushBackParallelTracks(std::vector<segment> &new_sol,
                                             const std::vector<segment> &replacementSet, direction firstDirection) {
    direction current_direction = firstDirection;
    for (auto s : replacementSet){
        segment s_hat = s;
        geometry.correct_segment_direction(s_hat, current_direction);
        new_sol.push_back(s_hat);
        current_direction = flip_direction(current_direction);
    }
}
