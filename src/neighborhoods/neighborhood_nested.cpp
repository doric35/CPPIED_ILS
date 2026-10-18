#include "../../include/neighborhoods/neighborhood_nested.hpp"

bool neighborhood_nested::local_search() {
    bool improved_solution = false;
    setRemovalGains();
    OneTwoReallocationMove move = getRandomMove();
    if (path_engine::is_node(move.replacements[0].source)) {
        applyRandomMove(move);
        n21_search();
        if (trial.cost < solution.cost) {
            solution          = trial;
            improved_solution = true;
        }
    }
    return improved_solution;
}

void neighborhood_nested::setRemovalGains() {
    removalGains.reserve(trial.path.size());
    for (auto it = std::next(trial.path.begin()); it != trial.path.end(); ++it)
        removalGains.push_back(geometry.removal_gain(trial, it));
}

OneTwoReallocationMove neighborhood_nested::getRandomMove() {
    randomMoveIndex = ris_heuristic::softmaxSample(removalGains, rng);
    randomMoveIndex++;
    RandomOneTwoReallocation random_allocation(problem, coverage, geometry, trial, randomMoveIndex);
    OneTwoReallocationMove move = random_allocation.solve();
    return move;
}

void neighborhood_nested::applyRandomMove(OneTwoReallocationMove &move) {
    coverage.remove(trial, trial.path[randomMoveIndex]);
    trial.path.erase(trial.path.begin() + randomMoveIndex);

    int first_offset = -1 * static_cast<int>(move.reallocationIndexes[0] > randomMoveIndex);
    int second_offset = -1 * static_cast<int>(move.reallocationIndexes[1] > randomMoveIndex);

    if (path_engine::is_node(move.replacements[0].source)){
        trial.path.insert(trial.path.begin() + move.reallocationIndexes[0] + first_offset, move.replacements[0]);
        coverage.insert(trial, move.replacements[0]);
    }
    if (path_engine::is_node(move.replacements[1].source)) {
        trial.path.insert(trial.path.begin() + move.reallocationIndexes[1] + 1 + second_offset, move.replacements[1]);
        coverage.insert(trial, move.replacements[1]);
    }
    trial.cost -= move.gain;
}

void neighborhood_nested::n21_search() {
    neighborhood_n21 nbh(ctx, problem, trial);
    bool improved = true;
    while (improved)
        improved = nbh.local_search();
}
