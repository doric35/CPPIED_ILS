#include "../../include/neighborhoods/neighborhood_gtsp.hpp"

bool neighborhood_gtsp::local_search() {
    trial = solution;
    bool improved_solution = false;
    setReplacements();
    std::vector<int> selection = selectClusters();
    tsp::GTSPGraph gtsp_instance = buildGtspInstance(selection);
    tspEngine.gtsp_to_atsp(gtsp_instance);
    gtsp_instance.tour = std::move(tspEngine.lkh(gtsp_instance.cost, gtsp_instance.tour));
    transformTspTourToSolution(gtsp_instance);
    if (trial.cost < solution.cost) {
        solution = trial;
        improved_solution = true;
    }
    return improved_solution;
}

void neighborhood_gtsp::reset() {
    replacements.clear();
}

void neighborhood_gtsp::setReplacements() {
    reset();
    replacements.emplace_back();
    for (int i=1; i< trial.path.size(); i++)
        replacements.push_back(getReplacementBucket(i));
}

ReplacementBucket neighborhood_gtsp::getReplacementBucket(int index) {
    coverage.remove(trial, trial.path[index]);
    ReplacementBucket bucket;
    std::vector<int> unsatisfied_cells = getUnsatisfiedCells();
    bucket.replacements = getReplacements(unsatisfied_cells);

    const segment index_segment = trial.path[index];
    const segment reversed_index_segment = geometry.flip_segment(index_segment);
    std::erase_if(bucket.replacements,[&](const segment& s){
        return s == index_segment || s == reversed_index_segment; });

    filterReplacementBucket(bucket, unsatisfied_cells);
    bucket.gain = gain(trial, trial.path.begin() + index, bucket.size());
    coverage.insert(trial, trial.path[index]);
    return bucket;
}

void neighborhood_gtsp::filterReplacementBucket(ReplacementBucket &bucket, std::vector<int>& unsatisfiedCells) {
    auto unsat_insertion = [&](const segment& s){
        return !coverage.insertion_satisfies(trial, s, unsatisfiedCells);
    };
    std::erase_if(bucket.replacements, unsat_insertion);
}

std::vector<int> neighborhood_gtsp::getUnsatisfiedCells() {
    std::vector<int> unsatisfied;
    coverage.unsatisfied_cells(trial, unsatisfied);
    return unsatisfied;
}

std::vector<int> neighborhood_gtsp::selectClusters() {
    auto selection_view = std::views::iota(0, static_cast<int>(replacements.size()));
    auto selection = std::ranges::to<std::vector<int>>(selection_view);
    auto null_gain = [&](int i ){
        return replacements[i].gain <= cost_t{0,0};
    };
    std::erase_if(selection, null_gain);
    RandomSegmentPacking selection_heuristic(problem, coverage, geometry, trial, selection);
    return selection_heuristic.solve();
}

std::vector<segment> neighborhood_gtsp::getReplacements(std::vector<int>& unsatisfiedCells) {
    iRectangle box;
    coverage.unsatisfied_rectangle(unsatisfiedCells, box);
    return getCompleteCoverageSegments(box);
}

tsp::GTSPGraph neighborhood_gtsp::buildGtspInstance(std::vector<int> &selection) {
    std::sort(selection.begin(), selection.end());
    tsp::GTSPGraph instance;
    buildDummyClusters(instance);
    for (int i=0, j=0; i< trial.path.size(); ++i){
        addSegmentClusters(instance);
        addSegmentToClusters(instance, trial.path[i]);
        if (j < selection.size() && i == selection[j]){
            addReplacementsToClusters(instance, i);
            ++j;
        }
        addClustersNodesToTour(instance);
    }
    instance.cost.resize(int(instance.nodes.size()),int(instance.nodes.size()));
    instance.cost.setConstant(trial.cost.length);
    instance.penalty = trial.cost.length + 1;
    setGtspCosts(instance);
    instance.cost.triangularView<Eigen::Lower>() = instance.cost.transpose();
    return instance;
}

void neighborhood_gtsp::buildDummyClusters(tsp::GTSPGraph& emptyInstance) {
    emptyInstance.tour.push_back(1);
    emptyInstance.tour.push_back(2);
    emptyInstance.tour.push_back(3);
    emptyInstance.nodes.push_back({0, 0, {path_engine::NULL_NODE, path_engine::NULL_NODE}});
    emptyInstance.nodes.push_back({1, 1, {path_engine::NULL_NODE, path_engine::NULL_NODE}});
    emptyInstance.nodes.push_back({2, 2, {path_engine::NULL_NODE, path_engine::NULL_NODE}});
    emptyInstance.clusters.push_back({0, {0}});
    emptyInstance.clusters.push_back({1, {1}});
    emptyInstance.clusters.push_back({2, {2}});
}

void neighborhood_gtsp::addSegmentClusters(tsp::GTSPGraph& instance) {
    for (int k=0; k<3; k++)
        instance.clusters.push_back({int(instance.clusters.size()), {}});

}

void neighborhood_gtsp::addSegmentToClusters(tsp::GTSPGraph& instance, segment& s) {
    int last_node = instance.nodes.back().id;
    std::array<segment, 3> seg_nodes = segment_to_nodes(s);
    instance.nodes.push_back({last_node+1, int(instance.clusters.size())-3, seg_nodes[0]});
    std::prev(instance.clusters.end(), 3)->nodes.push_back(last_node+1);

    instance.nodes.push_back({last_node+2, int(instance.clusters.size())-2,seg_nodes[1]});
    std::prev(instance.clusters.end(), 2)->nodes.push_back(last_node+2);

    instance.nodes.push_back({last_node+3,int(instance.clusters.size())-1,seg_nodes[2]});
    std::prev(instance.clusters.end(), 1)->nodes.push_back(last_node+3);
}

void neighborhood_gtsp::addReplacementsToClusters(tsp::GTSPGraph &instance, int index) {
    for (auto s : replacements[index])
        addSegmentToClusters(instance, s);
}

void neighborhood_gtsp::addClustersNodesToTour(tsp::GTSPGraph &instance) {
    for (auto n : std::prev(instance.clusters.end(), 3)->nodes)
        instance.tour.push_back(n+1);

    for (auto n : std::prev(instance.clusters.end(), 2)->nodes)
        instance.tour.push_back(n+1);

    for (auto n : std::prev(instance.clusters.end(), 1)->nodes)
        instance.tour.push_back(n+1);
}

void neighborhood_gtsp::setGtspCosts(tsp::GTSPGraph &instance) {
    setRootCosts(instance);
    for (int i=6; i< instance.nodes.size(); i+=3){
        setNodeSegmentCosts(instance, i);
        setForwardNodesCosts(instance, i);
    }
}

void neighborhood_gtsp::setRootCosts(tsp::GTSPGraph &instance) {
    segment flipped_first = geometry.flip_segment(instance.nodes[5].s);
    instance.cost(0, 1) = 0;
    instance.cost(1, 2) = 0;
    instance.cost(2,3) = 0;
    instance.cost(3,4) = 0;
    instance.cost(4,5) = 0;
    instance.cost(0, 5) = 0;
    for (int j=6; j<instance.nodes.size(); j+=3){
        instance.cost(0, j) = 0;
        instance.cost(0, j+2) = 0;
        instance.cost(5, j) = geometry.dist(flipped_first, instance.nodes[j].s).length;
        instance.cost(5, j+2) = geometry.dist(flipped_first, instance.nodes[j+2].s).length;
    }
}

void neighborhood_gtsp::setNodeSegmentCosts(tsp::GTSPGraph &instance, int node) {
    segment sf1 = geometry.flip_segment(instance.nodes[node].s);
    segment sf2 = geometry.flip_segment(instance.nodes[node+2].s);
    segment s = {sf1.source, sf2.source};
    if (s.source == s.target) s.target = path_engine::NULL_NODE;
    double s_cost = static_cast<double>(geometry.cost(s).length);
    instance.cost(node, node+1) = static_cast<int>(std::floor(s_cost / 2.0));
    instance.cost(node+1, node+2) = static_cast<int>(std::ceil(s_cost / 2.0));
}

void neighborhood_gtsp::setForwardNodesCosts(tsp::GTSPGraph &instance, int node) {
    segment sf1 = geometry.flip_segment(instance.nodes[node].s);
    segment sf2 = geometry.flip_segment(instance.nodes[node+2].s);
    for (int j=node+3; j<instance.nodes.size(); j+=3){
        if (instance.nodes[node].cluster_id / 3 != instance.nodes[j].cluster_id / 3){
            instance.cost(node, j) =
                    geometry.dist(sf1, instance.nodes[j].s).length;
            instance.cost(node, j+2) =
                    geometry.dist(sf1, instance.nodes[j+2].s).length;
            instance.cost(node+2,j) =
                    geometry.dist(sf2, instance.nodes[j].s).length;
            instance.cost(node+2, j+2) =
                    geometry.dist(sf2, instance.nodes[j+2].s).length;
        }
    }
}

void neighborhood_gtsp::transformTspTourToSolution(tsp::GTSPGraph &instance) {
    orderTspTour(instance);
    trial.path.clear();
    for(int i=3; i < instance.tour.size();){
        addClusterGroupSegmentToTrial(instance, instance.tour[i]);
        //Cluster group nodes must be traversed sequentially. The offset to the next group is exactly the three cluster
        //number of nodes.
        int cluster_size = getClusterGroupSize(instance, instance.tour[i]);
        i += cluster_size;
    }
    coverage.reset(trial);
    trial.cost = geometry.cost(trial);
}

void neighborhood_gtsp::orderTspTour(tsp::GTSPGraph &instance) {
    auto first_elem = std::find(instance.tour.begin(), instance.tour.end(), 1);
    std::rotate(instance.tour.begin(), first_elem, instance.tour.end());
    if (instance.tour.size() > 1 && instance.tour[1] != 2)
        std::reverse(std::next(instance.tour.begin()), instance.tour.end());
}

int neighborhood_gtsp::getClusterGroupSize(tsp::GTSPGraph& instance, int node) {
    const int node_cluster = instance.nodes[node - 1].cluster_id;
    int node_count = static_cast<int>(instance.clusters[node_cluster].nodes.size());
    if (node_cluster % 3 == 0)
        node_count += static_cast<int>(instance.clusters[node_cluster+1].nodes.size() + instance.clusters[node_cluster+2].nodes.size());
    else if (node_cluster % 3 == 1)
        node_count += static_cast<int>(instance.clusters[node_cluster+1].nodes.size() + instance.clusters[node_cluster-1].nodes.size());
    else
        node_count += static_cast<int>(instance.clusters[node_cluster-1].nodes.size() + instance.clusters[node_cluster-2].nodes.size());
    return node_count;
}

void neighborhood_gtsp::addClusterGroupSegmentToTrial(tsp::GTSPGraph &instance, int node) {
    int offset = -1 + 2*(node % 3 == 1) - 2*(node % 3 == 0);
    segment sf1 = instance.nodes[node-1].s;
    segment sf2 = instance.nodes[node+offset].s;
    if (sf1.source == sf2.source)
        trial.path.push_back(sf1);
    else
        trial.path.push_back(segment{sf1.source, sf2.source});
}

std::array<segment, 3> neighborhood_gtsp::segment_to_nodes(const segment& s){
    std::array<segment, 3> nodes{};
    direction dir = geometry.get_direction(s);
    direction other_dir = flip_direction(dir);
    nodes[0] = {s.source, path_engine::NULL_NODE};
    nodes[1] = {path_engine::NULL_NODE, path_engine::NULL_NODE};
    nodes[2] = {s.target, path_engine::NULL_NODE};
    if (!path_engine::is_node(s.target)) nodes[2].source = s.source;
    geometry.correct_segment_direction(nodes[0], dir);
    geometry.correct_segment_direction(nodes[2], other_dir);
    return nodes;
}

cost_t neighborhood_gtsp::gain(const cppied_solution &pSol,
                               neighborhood::sVecIt s,
                               int M) {
    cost_t r_gain = geometry.removal_gain(pSol, s);
    auto reallocable_gain = cost_t{0,1};
    reallocable_gain = reallocable_gain * int(M>0);
    return (r_gain + reallocable_gain)*int(M > 0);
}