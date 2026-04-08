#include "../../include/neighborhoods/neighborhood_gtsp.hpp"

bool neighborhood_gtsp::local_search(cppied_solution &pSol) {
    std::vector<int> selection;
    std::vector<int> warm_start;
    std::vector<std::vector<segment>> replacements;
    cluster_selection(pSol, selection, replacements);
    warm_start.reserve(pSol.path.size()*3 + 3);

    tsp::GTSPGraph graph;
    build_clusters(pSol,
                   graph,
                   replacements,
                   selection,
                   warm_start);

    assert(warm_start.size() == graph.nodes.size());
    graph.cost.resize(int(graph.nodes.size()),int(graph.nodes.size()));
    graph.cost.setConstant(pSol.cost.length);
    graph.penalty = pSol.cost.length + 1;

    set_gtsp_costs(graph);
    tspEngine.gtsp_to_atsp(graph);

    std::vector<int> tour = std::move(tspEngine.lkh(graph.cost, warm_start));
    cppied_solution trial = pSol;
    tsp_to_path(trial, tour, graph);
    trial.cost = geometry.cost(trial);
    if (trial.cost < pSol.cost) {
        pSol = std::move(trial);
        return true;
    }
    return false;
}

void neighborhood_gtsp::cluster_selection(cppied_solution &pSol,
                                          std::vector<int> &selection,
                                          std::vector<std::vector<segment>>& replacements) {
    std::vector<cost_t> gains;
    Eigen::VectorXi mask(pSol.coverage.size());
    mask.setZero();

    gains.resize(pSol.path.size(), cost_t{0,0});
    replacements.resize(pSol.path.size(), {});
    replacements[0] = {pSol.path[0]};
    for (int i=1; i< pSol.path.size(); i++){
        replacements[i] = {pSol.path[i]};
        add_replacements(pSol,replacements[i]);
        int M = static_cast<int>(replacements[i].size() - 1);
        gains[i] = gain(pSol, std::next(pSol.path.cbegin(), i), M);
    }

    auto best = std::max_element(gains.begin(), gains.end());
    if (*best <= cost_t{0,0})
        return;

    cost_t bound = {0,0};
    auto erase_func = [&] (int i){
        return gains[i] <= bound;
    };

    auto filter = [&](std::vector<int>& candidates){
        std::erase_if(candidates, erase_func);
    };

    std::vector<int> random_candidates(pSol.path.size(),0);
    std::iota(random_candidates.begin(), random_candidates.end(), 0);

    filter(random_candidates);
    std::shuffle(random_candidates.begin(), random_candidates.end(), rng);

    for (int i : random_candidates){
        if (!coverage.intersect_mask(pSol, pSol.path[i], mask)) {
            selection.push_back(i);
            coverage.mask(pSol, pSol.path[i], mask);
        }
    }
}

void neighborhood_gtsp::add_replacements(cppied_solution& pSol,
                                         std::vector<segment> &candidates) {
    direction dir = geometry.get_direction(candidates.front());
    coverage.remove(pSol, candidates.front());
    std::vector<int> unsat;
    coverage.unsatisfied_cells(pSol, candidates.front(), unsat);
    iRectangle box = {
            {std::numeric_limits<int>::max(),
                    std::numeric_limits<int>::max()},
            {std::numeric_limits<int>::min(),
                    std::numeric_limits<int>::min()}
    };
    coverage.unsatisfied_rectangle(unsat, box);
    auto segment_select = [&](segment& s){
        geometry.correct_segment_direction(s, dir);
        if (coverage.insertion_satisfies(pSol, s, unsat))
            candidates.push_back(s);
    };

    for_each_segment(box, candidates.front(), segment_select);

    coverage.insert(pSol, candidates.front());
}

void neighborhood_gtsp::set_gtsp_costs(tsp::GTSPGraph &graph) {
    //Root costs
    segment flipped_first = geometry.flip_segment(graph.nodes[5].s);
    graph.cost(0, 1) = 0;
    graph.cost(1, 2) = 0;
    graph.cost(2,3) = 0;
    graph.cost(3,4) = 0;
    graph.cost(4,5) = 0;
    graph.cost(0, 5) = 0;
    for (int j=6; j<graph.nodes.size(); j+=3){
        graph.cost(0, j) = 0;
        graph.cost(0, j+2) = 0;
        graph.cost(5, j) = geometry.dist(flipped_first, graph.nodes[j].s).length;
        graph.cost(5, j+2) = geometry.dist(flipped_first, graph.nodes[j+2].s).length;
    }
    //general_costs
    for (int i=6; i<graph.nodes.size(); i+=3){
        segment sf1 = geometry.flip_segment(graph.nodes[i].s);
        segment sf2 = geometry.flip_segment(graph.nodes[i+2].s);
        graph.cost(i, i+1) = 0;
        graph.cost(i+1, i+2) = 0;
        for (int j=i+3; j<graph.nodes.size(); j+=3){
            if (graph.nodes[i].cluster_id / 3 != graph.nodes[j].cluster_id / 3){
                graph.cost(i, j) =
                        geometry.dist(sf1, graph.nodes[j].s).length;
                graph.cost(i, j+2) =
                        geometry.dist(sf1, graph.nodes[j+2].s).length;
                graph.cost(i+2,j) =
                        geometry.dist(sf2, graph.nodes[j].s).length;
                graph.cost(i+2, j+2) =
                        geometry.dist(sf2, graph.nodes[j+2].s).length;
            }
        }
    }
}

void neighborhood_gtsp::build_clusters(const cppied_solution &pSol,
                                       tsp::GTSPGraph &target,
                                       std::vector<std::vector<segment>>& replacements,
                                       std::vector<int>& selection,
                                       std::vector<int>& warm_start) {
    std::sort(selection.begin(), selection.end());

    //Build the dummy clusters
    warm_start.push_back(1);
    warm_start.push_back(2);
    warm_start.push_back(3);
    target.nodes.push_back({0, 0, {NULL_NODE, NULL_NODE}});
    target.nodes.push_back({1, 1, {NULL_NODE, NULL_NODE}});
    target.nodes.push_back({2, 2, {NULL_NODE, NULL_NODE}});
    target.clusters.push_back({0, {0}});
    target.clusters.push_back({1, {1}});
    target.clusters.push_back({2, {2}});

    int j=0;
    int node_id = 3;
    for (int i=0; i< pSol.path.size(); i++){
        std::vector<segment> neighbors{pSol.path[i]};
        if (j < selection.size() && i==selection[j]) {
            assert(!replacements[i].empty());
            neighbors = replacements[i];
            ++j;
        }
        for (int k=0; k<3; k++)
            target.clusters.push_back({int(target.clusters.size()), {}});

        int first_node_id = node_id;

        for (auto& nb: neighbors){
            std::array<segment, 3> seg_nodes = segment_to_nodes(nb);
            target.nodes.push_back({node_id,
                                    int(target.clusters.size())-3,
                                    seg_nodes[0]});
            std::prev(target.clusters.end(), 3)->nodes.push_back(node_id++);

            target.nodes.push_back({node_id,
                                    int(target.clusters.size())-2,
                                    seg_nodes[1]});
            std::prev(target.clusters.end(), 2)->nodes.push_back(node_id++);

            target.nodes.push_back({node_id,
                                    int(target.clusters.size())-1,
                                    seg_nodes[2]});
            std::prev(target.clusters.end(), 1)->nodes.push_back(node_id++);
        }
        for (auto n : std::prev(target.clusters.end(), 3)->nodes)
            warm_start.push_back(n+1);

        for (auto n : std::prev(target.clusters.end(), 2)->nodes)
            warm_start.push_back(n+1);

        for (auto n : std::prev(target.clusters.end(), 1)->nodes)
            warm_start.push_back(n+1);
    }
}

void neighborhood_gtsp::tsp_to_path(cppied_solution &pSol,
                                   std::vector<int>& tour,
                                   tsp::GTSPGraph& graph) {
    auto first_elem = std::find(tour.begin(), tour.end(), 1);

    if (first_elem != std::prev(tour.end())
        && *std::next(first_elem)!=2)
        std::reverse(tour.begin(), tour.end());
    first_elem = std::find(tour.begin(), tour.end(), 1);
    if (first_elem != tour.begin())
        std::rotate(tour.begin(), first_elem, tour.end());

    std::vector<bool> visited(graph.clusters.size(), false);
    pSol.path.clear();
    std::vector<segment> stack;
    stack.reserve(3);

    assert(tour[0] == 1 && tour[1] == 2 && tour[2] == 3);
    for (int i=3; i<tour.size(); i++){
        if (tour[i] - 1 < 0 || tour[i] - 1 >= static_cast<int>(graph.nodes.size()))
            throw std::out_of_range(
                "TSP tour index " + std::to_string(tour[i]) +
                " out of range [1, " + std::to_string(graph.nodes.size()) +
                "]; a stale solution file may have been read.");
        int c = graph.nodes[tour[i]-1].cluster_id;
        if (!visited[c]){
            stack.push_back(graph.nodes[tour[i]-1].s);
            if (stack.size() == 3){
                segment s{};
                try {
                    s = nodes_to_segment(stack);
                } catch(std::runtime_error& e) {
                    for (auto& seg : pSol.path)
                        std::cerr << seg << std::endl;
                    std::cerr << e.what() << std::endl;
                    throw e;
                }
                pSol.path.push_back(s);
                stack.clear();
            }
            visited[c] = true;
        }
    }
    coverage.reset(pSol);

}

std::array<segment, 3> neighborhood_gtsp::segment_to_nodes(const segment& s){
    std::array<segment, 3> nodes{};
    direction dir = geometry.get_direction(s);
    direction other_dir = flip_direction(dir);
    nodes[0] = {s.source, NULL_NODE};
    nodes[1] = {NULL_NODE, NULL_NODE};
    nodes[2] = {s.target, NULL_NODE};
    if (!is_node(s.target)) nodes[2].source = s.source;
    geometry.correct_segment_direction(nodes[0], dir);
    geometry.correct_segment_direction(nodes[2], other_dir);
    return nodes;
}

segment neighborhood_gtsp::nodes_to_segment(const std::vector<segment> &nodes) {
    if (!is_node(nodes[0].source) || !is_node(nodes[2].source))
        throw std::runtime_error("Found NULL segment encoded in a source.");
    if (nodes[0].source == nodes[2].source)
        return nodes[0];
    return segment{nodes[0].source, nodes[2].source};
}

cost_t neighborhood_gtsp::gain(const cppied_solution &pSol,
                               neighborhood::sVecIt s,
                               int M) {
    cost_t r_gain = geometry.removal_gain(pSol, s);
    auto reallocable_gain = cost_t{0,1};
    reallocable_gain = reallocable_gain * int(M>0);
    return (r_gain * int(M>0)) + reallocable_gain;
}