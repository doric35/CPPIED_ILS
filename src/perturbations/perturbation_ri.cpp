#include "../../include/perturbations/perturbation_ri.hpp"

void perturbation_ri::d_perturbate() {
    insertionSample.clear();
    insertionSample.reserve(sampleSize);
    sampleCandidateUnitarySegments();
    insertCandidates();
}

void perturbation_ri::sampleCandidateUnitarySegments() {
    std::vector<double> weights = getCandidateWeights();
    std::vector<int> sorted_candidates = getSortedCandidates(weights);
    setCandidateSampling(sorted_candidates);
}

std::vector<double> perturbation_ri::getCandidateWeights() {
    std::vector<double> weights(problem.vertex.size(), 0.0);
    for (int v=0; v < weights.size(); v++)
        weights[v] = sampleCandidateWeight(v);
    return weights;
}

double perturbation_ri::sampleCandidateWeight(int u) {
    segment dummy = {u, path_engine::NULL_NODE};
    double oc = coverage.over_coverage(solution, dummy);
    double w = dist(rng);
    return std::log(w) / (oc + EPSILON);
}

std::vector<int> perturbation_ri::getSortedCandidates(
    const std::vector<double> &weights
) const {
    std::vector<int> candidates(weights.size());
    std::iota(candidates.begin(), candidates.end(), 0);

    auto cmpWeights = [&](int a, int b) {
        return weights[a] > weights[b];
    };

    std::nth_element(
        candidates.begin(),
        candidates.begin() + static_cast<int>(sampleSize),
        candidates.end(),
        cmpWeights
    );

    return candidates;
}

void perturbation_ri::setCandidateSampling(const std::vector<int> &partiallySortedCandidates) {
    for (std::size_t i=0; i<sampleSize; ++i)
        insertionSample.push_back({partiallySortedCandidates[i], path_engine::NULL_NODE});
}

void perturbation_ri::insertCandidates() {
    std::shuffle(insertionSample.begin(), insertionSample.end(), rng);
    for (auto s : insertionSample)
        insertCandidate(s);
}

void perturbation_ri::insertCandidate(segment &s) {
    MinimumSegmentInsertion insertion_algorithm(problem, coverage, geometry, solution, s);
    SegmentInsertion s_hat = insertion_algorithm.solve();
    solution.path.insert(solution.path.begin() + s_hat.index, s_hat.s);
    solution.cost += s_hat.insertion_cost;
    coverage.insert(solution, s_hat.s);
}
