#include "../../include/neighborhoods/neighborhood_tsp.hpp"

bool neighborhood_tsp::local_search() {
    bool improved = false;
    trial = solution;
    splitTrial();
    TspInstance instance = buildTspInstance();
    moveTrialFromTspSolution(instance);
    if (trial.cost < solution.cost) {
        solution = trial;
        improved = true;
    }
    return improved;
}

TspInstance neighborhood_tsp::buildTspInstance() {
    TspInstance instance{};
    instance.distances = getDistanceMatrix();
    auto tsp_solution_view = std::views::iota(1, tspDimension+1);
    instance.tour = std::ranges::to<std::vector<int>>(tsp_solution_view);
    return instance;
}

Eigen::MatrixXi neighborhood_tsp::getDistanceMatrix() {
    Eigen::MatrixXi C(tspDimension, tspDimension);
    C.setConstant(trial.cost.length);
    connectDummyNodes(C);
    connectSegmentsToEndNode(C);
    connectPairWiseSegments(C);
    C.triangularView<Eigen::Lower>() = C.transpose();
    return C;
}

void neighborhood_tsp::splitTrial() {
    std::vector<int> split_selection = selectSplitIndexes();
    std::vector<segment> path = constructSplitPath(split_selection);
    trial.path.swap(path);
    tspDimension = static_cast<int>(trial.path.size()) * 3 + 3;
}

std::vector<segment> neighborhood_tsp::constructSplitPath(std::vector<int> &splitSelection) {
    std::vector<segment> path;
    path.reserve(trial.path.size() + splitSelection.size());
    for (int i=0, j=0; i<trial.path.size(); i++){
        if ( j < splitSelection.size() && i == splitSelection[j] ) {
            splitAndPush(path, trial.path[i]);
            j++;
        } else
            path.push_back(trial.path[i]);
    }
    return path;
}

void neighborhood_tsp::splitAndPush(std::vector<segment> &partialPath, const segment &s) {
    int                         index = getSplitIndex(s);
    std::pair<segment, segment> split = geometry.splitSegment(s, index);
    partialPath.push_back(split.first);
    if (path_engine::is_node(split.second.source)) partialPath.push_back(split.second);
}

std::vector<int> neighborhood_tsp::selectSplitIndexes(){
    const int n_splits = int(std::sqrt(static_cast<double>(trial.path.size())));
    auto selection_view = std::views::iota(0, static_cast<int>(trial.path.size()));
    auto selection = std::ranges::to<std::vector<int>>(selection_view);
    auto cmp = [&](int i, int j){
        return geometry.cost(trial.path[i]) > geometry.cost(trial.path[j]);
    };
    std::partial_sort(selection.begin(),std::next(selection.begin(), n_splits),
                      selection.end(), cmp);
    std::vector<int> split_selection(selection.begin(),
                                     std::next(selection.begin(), n_splits));
    std::sort(split_selection.begin(), split_selection.end());
    return split_selection;
}

int neighborhood_tsp::getSplitIndex(const segment& s) {
    cost_t cost = geometry.cost(s);
    int min_cut = 0;
    int max_cut = std::max(0, cost.length - 1);
    std::uniform_int_distribution<> d(min_cut, max_cut);
    int index = d(rng);
    return index;
}

void neighborhood_tsp::moveTrialFromTspSolution(TspInstance &instance) {
    instance.tour = std::move(tspEngine.lkh(instance.distances,instance.tour));
    setTrialFromTspTour(instance.tour);
    trial.cost = geometry.cost(trial);
}

void neighborhood_tsp::connectDummyNodes(Eigen::Ref<Eigen::MatrixXi> distances) {
    distances(0,1) = 0;
    distances(1,2) = 0;
    distances(2, 3) = 0;
}

void neighborhood_tsp::connectSegmentsToEndNode(Eigen::Ref<Eigen::MatrixXi> distances) const {
    for (int i=1; i<trial.path.size(); i++){
        int mat_tsp_idx = (i+1) * 3;
        distances(0, mat_tsp_idx) = 0;
        distances(0, mat_tsp_idx+2) =0;
    }
}

void neighborhood_tsp::connectPairWiseSegments(const Eigen::Ref<Eigen::MatrixXi>& distances) {
    for (int i =0; i< trial.path.size(); i++){
        connectSegmentNodes(distances, i);
        connectSegmentToForwardNodes(distances, i);
    }
}

void neighborhood_tsp::connectSegmentToForwardNodes(const Eigen::Ref<Eigen::MatrixXi>& distances, int path_index) {
    for (int j = path_index + 1; j < trial.path.size(); j++)
        connectSegmentPair(distances, path_index, j);
}

void neighborhood_tsp::connectSegmentNodes(Eigen::Ref<Eigen::MatrixXi> distances, int path_index) {
    int tsp_idx = (path_index + 1) * 3;
    distances(tsp_idx, tsp_idx+1) = 0;
    distances(tsp_idx+1, tsp_idx+2) = 0;
}

void neighborhood_tsp::connectSegmentPair(Eigen::Ref<Eigen::MatrixXi> distances, int first, int second) {
    int source_tsp_index = (first + 1) * 3;
    int target_tsp_index = (second + 1) * 3;
    const segment first_segment = trial.path[first];
    const segment first_flipped = geometry.flip_segment(first_segment);
    const segment second_segment = trial.path[second];
    const segment second_flipped = geometry.flip_segment(second_segment);
    if (first > 0){
        distances(source_tsp_index, target_tsp_index)  = geometry.dist(first_flipped, second_segment).length;
        distances(source_tsp_index, target_tsp_index+2) = geometry.dist(first_flipped, second_flipped).length;
    }
    distances(source_tsp_index + 2, target_tsp_index) = geometry.dist(first_segment, second_segment).length;
    distances(source_tsp_index + 2, target_tsp_index + 2) = geometry.dist(first_segment, second_flipped).length;
}

void neighborhood_tsp::setTrialFromTspTour(std::vector<int> &tour) {
    orderTspTourAndOrientation(tour);

    //Positions 0, 1, and 2 of the tour are now the dummy nodes.
    std::vector<segment> path;
    for (int i=3; i<tour.size(); i+=3){
        segment s = tour_id_to_segment(tour[i]);
        path.push_back(s);
    }
    trial.path.swap(path);
}

void neighborhood_tsp::orderTspTourAndOrientation(std::vector<int> &unorderedTour) {
    auto first_elem = std::find(unorderedTour.begin(), unorderedTour.end(), 1);
    std::rotate(unorderedTour.begin(), first_elem, unorderedTour.end());
    if (unorderedTour.size() > 1 && unorderedTour[1] != 2)
        std::reverse(std::next(unorderedTour.begin()), unorderedTour.end());
}

segment neighborhood_tsp::tour_id_to_segment(int i) {
    int idx;
    segment s{};
    if (i % 3 == 1) {
        idx = (i - 2) / 3;
        s = trial.path[idx];
    } else if (i % 3 == 0) {
        idx = (i - 6) / 3;
        s = geometry.flip_segment(trial.path[idx]);
    } else {
        throw std::runtime_error("Encountered segment mid point in tsp solution.");
    }

    return s;
}
