#include "../../include/perturbations/perturbation_db.hpp"

void perturbation_db::d_perturbate() {
    if (isValidSolutionInstance(solution)) {
        std::array<int, 4> db_sample = getDoubleBridgeIndexes();
        applyDoubleBridgeMove(db_sample);
    }
}

std::array<int, 4> perturbation_db::getDoubleBridgeIndexes() {
    std::vector<int> candidates = uniformCandidateSample();
    int first_index = greedyCandidateSample(candidates);
    std::array<int, 6> top_six = topSixCandidates(candidates, first_index);
    int end_candidate_index = std::min(6, static_cast<int>(candidates.size()));
    std::shuffle(top_six.begin() + 1, top_six.begin() + end_candidate_index, rng);
    std::sort(top_six.begin(), top_six.begin() + 4);
    std::array<int, 4> db_sample;
    std::copy_n(top_six.begin(), 4, db_sample.begin());
    return db_sample;
}

std::vector<int> perturbation_db::uniformCandidateSample() {
    std::uniform_int_distribution<> d(0,1);
    int mod = d(rng);
    auto candidates_view = std::views::iota(1, static_cast<int>(solution.path.size())-1) |
                      std::views::filter([mod](int i){ return i % 2 == mod; });
    std::vector<int> candidates(candidates_view.begin(), candidates_view.end());
    std::shuffle(candidates.begin(), candidates.end(), rng);
    int k = std::max(std::log(solution.path.size() / 2.0), 8.0);
    k = std::min(k, static_cast<int>(candidates.size()));
    return std::vector<int>(candidates.begin(), candidates.begin() + k);
}

int perturbation_db::greedyCandidateSample(std::vector<int> &candidates) {
    std::vector<cost_t> gains;
    gains.reserve(candidates.size());
    auto func = [&](int i){ return gain(candidates, i); };
    std::transform(candidates.begin(), candidates.end(),
                   std::back_inserter(gains), func);
    auto elem = std::max_element(gains.begin(), gains.end());
    int sample = static_cast<int>(std::distance(gains.begin(), elem));
    return candidates[sample];
}

std::array<int, 6> perturbation_db::topSixCandidates(std::vector<int> &candidates, int ref) {
    std::array<int, 6> top_six{0,0,0,0,0,0};
    std::vector<cost_t> costs = getDistancesToReferenceCandidate(candidates, ref);
    std::vector<int> sample = getOrderedIndexes(costs);

    for ( int i=0; i< std::min(6, int(sample.size())); i++)
        top_six[i] = candidates[sample[i]];

    return top_six;
}

std::vector<cost_t> perturbation_db::getDistancesToReferenceCandidate(std::vector<int> &candidates, int ref) {
    std::vector<cost_t> costs;
    costs.reserve(candidates.size());

    auto func = [&](int i){
        if (i==ref)
            return cost_t{0,0};
        return geometry.dist(solution.path[ref], solution.path[i]);
    };

    std::transform(candidates.begin(), candidates.end(),
                   std::back_inserter(costs), func);
    return costs;
}

std::vector<int> perturbation_db::getOrderedIndexes(std::vector<cost_t> &distances) {
    auto candidates_view = std::views::iota(0, static_cast<int>(distances.size()));
    std::vector<int> candidates(candidates_view.begin(), candidates_view.end());
    auto cmp = [&](int i, int j){
        return distances[i] < distances[j];
    };
    std::sort(candidates.begin(), candidates.end(), cmp);
    return candidates;
}

cost_t perturbation_db::gain(std::vector<int> &sampling,
                             int i) {
    cost_t best = {std::numeric_limits<int>::max(),
                   std::numeric_limits<int>::max()};
    cost_t base = geometry.dist(solution.path[i], solution.path[i+1]);
    for (int j : sampling){
        if (i != j && base + geometry.dist(solution.path[i],solution.path[j]) < best)
            best = base + geometry.dist(solution.path[i],solution.path[j]);

    }
    return best;
}

void perturbation_db::applyDoubleBridgeMove(std::array<int, 4> &selection) {
    std::vector<segment> new_sol;
    new_sol.reserve(2*solution.path.size());
    for (int i=0; i <= selection[0]; i++)
        new_sol.push_back(solution.path[i]);
    geometry.link(new_sol, solution.path[selection[2]+1]);
    for (int i=selection[2] + 2; i<= selection[3]; i++)
        new_sol.push_back(solution.path[i]);
    geometry.link(new_sol, solution.path[selection[1] + 1]);
    for (int i=selection[1] + 2; i<= selection[2]; i++)
        new_sol.push_back(solution.path[i]);
    geometry.link(new_sol, solution.path[selection[0] + 1]);
    for (int i=selection[0] + 2; i<= selection[1]; i++)
        new_sol.push_back(solution.path[i]);
    geometry.link(new_sol, solution.path[selection[3] + 1]);
    for (int i=selection[3] + 2; i< solution.path.size(); i++)
        new_sol.push_back(solution.path[i]);
    solution.path.swap(new_sol);
    solution.cost = geometry.cost(solution);
}
