#include "../test_fixture.hpp"
#include "../../include/neighborhoods/neighborhood_gtsp.hpp"

struct neighborhood_gtsp_accessor : public neighborhood_gtsp {
public:
    using neighborhood_gtsp::neighborhood_gtsp;

    using neighborhood::geometry;
    using neighborhood::coverage;
    using neighborhood::problem;

    using neighborhood_gtsp::tspEngine;
};

class neighborhood_gtsp_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<neighborhood_gtsp_accessor> n;
    cppied_solution sol;

    static constexpr const char* scratch_dir =
            "/home/doric35/projects/def-mmorin-ab/doric35/CppiedEjor/tests/scratch";

    void SetUp() override {
        cppied_context_fixture::SetUp();
        sol.path = {{6,  11},
                    {23, 18},
                    {44, 42},
                    {48, 51},
                    {30, 35},
                    {75, 73},
                    {62, 63}};
        sol.cost     = {0, 0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        n = std::make_unique<neighborhood_gtsp_accessor>(ctx, P);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }

    // Build a minimal, ready-to-use GTSPGraph from sol (no replacements).
    tsp::GTSPGraph make_graph(cppied_solution& s) {
        tsp::GTSPGraph g;
        std::vector<int> sel;
        std::vector<std::vector<segment>> rep;
        std::vector<int> ws;
        n->build_clusters(s, g, rep, sel, ws);
        g.cost.resize(int(g.nodes.size()), int(g.nodes.size()));
        g.cost.setConstant(s.cost.length);
        n->set_gtsp_costs(g);
        return g;
    }
};

// ============================================================================
// segment_to_nodes / nodes_to_segment
// ============================================================================

// Each segment maps to exactly 3 TSP nodes: [entry, dummy_middle, reversed_exit].
TEST_F(neighborhood_gtsp_fixture, SegmentToNodesReturnsSizeThree) {
    auto nodes = n->segment_to_nodes(sol.path[0]);
    EXPECT_EQ(nodes.size(), 3u);
}

// nodes[0] is the entry node: same source as the segment, same travel direction.
TEST_F(neighborhood_gtsp_fixture, SegmentToNodesEntrySourceMatchesSegment) {
    segment s = {6, 11};
    auto nodes = n->segment_to_nodes(s);
    EXPECT_EQ(nodes[0].source, s.source);
    EXPECT_EQ(n->geometry.get_direction(nodes[0]),
              n->geometry.get_direction(s));
}

// nodes[2] is the reversed-exit node: source is the segment target, direction
// is the opposite of the segment's travel direction.
TEST_F(neighborhood_gtsp_fixture, SegmentToNodesExitSourceMatchesTarget) {
    segment s = {6, 11};
    auto nodes = n->segment_to_nodes(s);
    direction fwd   = n->geometry.get_direction(s);
    direction rev   = flip_direction(fwd);
    EXPECT_EQ(nodes[2].source, s.target);
    EXPECT_EQ(n->geometry.get_direction(nodes[2]), rev);
}

// nodes[1] is the dummy middle: both fields must be path_engine::NULL_NODE.
TEST_F(neighborhood_gtsp_fixture, SegmentToNodesMiddleIsDummy) {
    auto nodes = n->segment_to_nodes(sol.path[0]);
    EXPECT_EQ(nodes[1].source, path_engine::NULL_NODE);
    EXPECT_EQ(nodes[1].target, path_engine::NULL_NODE);
}

// nodes_to_segment(segment_to_nodes(s)) must recover the original segment.
TEST_F(neighborhood_gtsp_fixture, NodesToSegmentRoundtripForward) {
    segment original = {6, 11};
    auto arr = n->segment_to_nodes(original);
    std::vector<segment> v(arr.begin(), arr.end());
    segment recovered = n->nodes_to_segment(v);
    EXPECT_EQ(recovered.source, original.source);
    EXPECT_EQ(recovered.target, original.target);
}

// Round-trip must also work for a reversed (W/N-direction) segment.
TEST_F(neighborhood_gtsp_fixture, NodesToSegmentRoundtripReversed) {
    segment original = {23, 18};   // W direction: source > target, both E-range
    auto arr = n->segment_to_nodes(original);
    std::vector<segment> v(arr.begin(), arr.end());
    segment recovered = n->nodes_to_segment(v);
    EXPECT_EQ(recovered.source, original.source);
    EXPECT_EQ(recovered.target, original.target);
}

// ============================================================================
// build_clusters — structure
// ============================================================================

// Each path segment contributes exactly 3 nodes to the GTSP graph.
TEST_F(neighborhood_gtsp_fixture, BuildClustersNodeCountThreePerSegment) {
    auto g = make_graph(sol);
    EXPECT_EQ(g.nodes.size(), sol.path.size() * 3 + 3);
}

// Each path segment creates 3 GTSP clusters (entry, middle, exit groups).
TEST_F(neighborhood_gtsp_fixture, BuildClustersClusterCountThreePerSegment) {
    auto g = make_graph(sol);
    EXPECT_EQ(g.clusters.size(), sol.path.size() * 3 + 3);
}

// warm_start has one entry per node in the graph.
TEST_F(neighborhood_gtsp_fixture, BuildClustersWarmStartSizeMatchesNodeCount) {
    tsp::GTSPGraph g;
    std::vector<int> sel;
    std::vector<std::vector<segment>> rep;
    std::vector<int> ws;
    n->build_clusters(sol, g, rep, sel, ws);
    EXPECT_EQ(ws.size(), g.nodes.size());
}

// CORRECTNESS CONTRACT: node IDs in graph.clusters must be 0-based so they
// can be used directly as matrix row/column indices in gtsp_to_sgtsp.
// Bug: build_clusters starts node_id at 1, making all cluster.nodes entries
// 1-based — off by one when used as Eigen matrix indices.
TEST_F(neighborhood_gtsp_fixture, BuildClustersNodeIDsAreZeroBased) {
    tsp::GTSPGraph g;
    std::vector<int> sel;
    std::vector<std::vector<segment>> rep;
    std::vector<int> ws;
    n->build_clusters(sol, g, rep, sel, ws);
    EXPECT_EQ(g.clusters[0].nodes[0], 0)
        << "cluster.nodes must hold 0-based matrix positions; "
           "currently node_id starts at 1 causing off-by-one in gtsp_to_sgtsp";
}

// Each node in graph.nodes at position i must carry id == i so that 1-based
// TSP tour indices (tour[k]-1) map back correctly to graph.nodes positions.
TEST_F(neighborhood_gtsp_fixture, BuildClustersNodeIDMatchesPosition) {
    tsp::GTSPGraph g;
    std::vector<int> sel;
    std::vector<std::vector<segment>> rep;
    std::vector<int> ws;
    n->build_clusters(sol, g, rep, sel, ws);
    for (int i = 0; i < static_cast<int>(g.nodes.size()); ++i)
        EXPECT_EQ(g.nodes[i].id, i)
            << "node at position " << i << " has id " << g.nodes[i].id
            << "; id must equal position for correct matrix/tour indexing";
}

// ============================================================================
// set_gtsp_costs — cost matrix content
// ============================================================================

// Intra-segment forward edges (position i→i+1 and i+1→i+2) must be zero
// for every segment triple starting at position 0, 3, 6, ...
TEST_F(neighborhood_gtsp_fixture, SetGtspCostsIntraSegmentEdgesAreZero) {
    auto g = make_graph(sol);
    for (int i = 0; i < static_cast<int>(g.nodes.size()) - 2; i += 3) {
        EXPECT_EQ(g.cost(i,   i+1), 0) << "intra edge (" << i   << "," << i+1 << ")";
        EXPECT_EQ(g.cost(i+1, i+2), 0) << "intra edge (" << i+1 << "," << i+2 << ")";
    }
}

// From the root (position 0) to every other segment's entry (j) and
// reversed-exit (j+2), the cost must be 0 (unconstrained ordering).
TEST_F(neighborhood_gtsp_fixture, SetGtspCostsRootToOtherSegmentsAreZero) {
    auto g = make_graph(sol);
    EXPECT_EQ(g.cost(0, 5),   0) << "root->entry  at position " << 5;
    for (int j = 6; j < static_cast<int>(g.nodes.size()) - 2; j += 3) {
        EXPECT_EQ(g.cost(0, j),   0) << "root->entry  at position " << j;
        EXPECT_EQ(g.cost(0, j+2), 0) << "root->r-exit at position " << j+2;
    }
}

// The cost from the exit of segment 0 (position 2) to the entry of segment j
// (position j) must equal geometry.dist(flip(seg0), seg_j).length.
TEST_F(neighborhood_gtsp_fixture, SetGtspCostsExitToEntryMatchesGeometry) {
    auto g = make_graph(sol);
    for (int j = 6; j < static_cast<int>(g.nodes.size()) - 2; j += 3) {
        int expected = n->geometry.dist(sol.path[0], g.nodes[j].s).length;
        EXPECT_EQ(g.cost(5, j), expected)
            << "exit(seg0)->entry(seg at pos " << j << "): "
               "expected " << expected << " got " << g.cost(5, j);
    }
}

// CORRECTNESS CONTRACT: for every pair of segments (i, j), the cost from
// the exit of segment i (position i+2) to the entry of segment j (position j)
// must be set to an actual geometry distance, not left at the sentinel MAX/2.
// Bug: cluster_id % 3 is always 0 for entry-type nodes of different segments,
// so the condition `cluster_id_i % 3 != cluster_id_j % 3` is always false and
// no cross-segment edge is ever written for same-role node pairs.
TEST_F(neighborhood_gtsp_fixture, SetGtspCostsCrossSegmentEdgesAreSet) {
    auto g = make_graph(sol);
    const int sentinel = std::numeric_limits<int>::max() / 2;
    for (int i = 0; i < static_cast<int>(g.nodes.size()) - 5; i += 3) {
        for (int j = i + 3; j < static_cast<int>(g.nodes.size()) - 2; j += 3) {
            EXPECT_NE(g.cost(i+2, j), sentinel)
                << "exit(pos " << i+2 << ") -> entry(pos " << j << ") was never set "
                   "(cluster_id % 3 condition is always false for same-role nodes)";
        }
    }
}

// ============================================================================
// gtsp_to_sgtsp — indexing correctness
// ============================================================================

// After gtsp_to_sgtsp, the cost matrix entry C(n0, n1) where n0 and n1 are
// the first two nodes of a cluster (0-based positions) must be 0 (or shifted
// by the sum penalty), indicating forced intra-cluster traversal.
// Bug: cluster.nodes stores 1-based IDs. C_hat(nodes[0], nodes[1]) sets
// C_hat(1, 2) instead of C_hat(0, 1), leaving C_hat(0, 1) = INT_MAX.
TEST_F(neighborhood_gtsp_fixture, GtspToSgtspIntraClusterEdgeAtPositionZeroIsSet) {
    auto g = make_graph(sol);
    n->tspEngine.gtsp_to_sgtsp(g);
    // After the transformation the intra-cluster edge at 0-based positions
    // (0,1) must not remain at the initial MAX value.
    EXPECT_NE(g.cost(0, 1), std::numeric_limits<int>::max())
        << "Intra-cluster edge at (0,1) is MAX after gtsp_to_sgtsp; "
           "1-based node IDs are used as 0-based matrix indices, "
           "so C_hat(1,2) is set instead of C_hat(0,1)";
}

// The last-node inter-cluster edges must be propagated from the correct row.
// Bug: in the loop `int r = clusters[i].nodes.size()-1; graph.cost(r, k_l)`
// uses r (a cluster-local index, e.g. 0 for a 1-element cluster) as a matrix
// row instead of graph.clusters[i].nodes[r] (the actual node matrix position).
TEST_F(neighborhood_gtsp_fixture, GtspToSgtspLastNodeInterClusterEdgesCorrect) {
    // Two-segment path for predictable node positions.
    cppied_solution two;
    two.path     = {{6, 11}, {30, 35}};
    two.coverage = Eigen::VectorXd::Zero(P.req.size());
    n->coverage.reset(two);
    two.cost = n->geometry.cost(two);

    auto g = make_graph(two);

    // Pre-transformation: record the cost that should flow from position 2
    // (exit of segment 0) to position 3 (entry of segment 1).
    int pre = g.cost(2, 3);

    n->tspEngine.gtsp_to_sgtsp(g);

    // After transformation, graph.cost(2, 3) must not be MAX: the
    // inter-cluster edge from the last node of cluster-group 0 (pos 2)
    // to the first node of cluster-group 1 (pos 3) must survive.
    // If 'r' bug is present, graph.cost(0, 3) is set instead and
    // graph.cost(2, 3) remains MAX.
    EXPECT_NE(g.cost(2, 3), std::numeric_limits<int>::max())
        << "Inter-cluster edge (2,3) lost after gtsp_to_sgtsp; "
           "r used as matrix row instead of cluster.nodes[r]";
    (void)pre;
}

TEST_F(neighborhood_gtsp_fixture, NoEmptyGtspClusters){
    auto g = make_graph(sol);
    for (auto & cluster : g.clusters)
        EXPECT_FALSE(cluster.nodes.empty());
}

// ============================================================================
// E2E — require LKH binary configured in test_config.txt.
//
// Contract of neighborhood_gtsp::local_search (mirrors neighborhood_tsp):
//   • Expects a non-completed, coverage-initialised survey-only path.
//   • On TRUE:  geometry.complete + trim + cut applied; cost < incumbent.
//   • On FALSE: path holds TSP-reordered non-completed segments.
//
// NOTE: cluster_selection contains `while (available.size() > 0)` which
// never terminates because available is never resized. If that bug is present
// the E2E tests below will hang. Fix the loop condition to
// `std::any_of(available.begin(), available.end(), std::identity{})` first.
// ============================================================================

class neighborhood_gtsp_e2e_fixture : public neighborhood_gtsp_fixture {
protected:
    void SetUp() override {
        neighborhood_gtsp_fixture::SetUp();
        std::filesystem::create_directories(scratch_dir);
        ctx.start_time = std::chrono::high_resolution_clock::now();
    }
};

// local_search must not throw on a valid, coverage-feasible input.
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchDoesNotThrow) {
    EXPECT_NO_THROW(n->local_search(sol));
}

// The path must be non-empty after the call regardless of return value.
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchPathNonEmpty) {
    n->local_search(sol);
    EXPECT_GT(sol.path.size(), 0u);
}

// On TRUE the tracked cost must be strictly less than the pre-call incumbent.
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchCostDecreasedWhenReturnTrue) {
    cost_t before = sol.cost;
    if (n->local_search(sol))
        EXPECT_LT(sol.cost, before);
}

// On TRUE the tracked cost must equal geometry.cost(sol) (trim+cut leave the
// cost field consistent).
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchCostConsistentWhenReturnTrue) {
    if (n->local_search(sol))
        EXPECT_EQ(n->geometry.cost(sol), sol.cost);
}

// On TRUE the coverage vector must match a fresh reset (incremental updates
// must agree with a full recompute).
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchCoverageConsistentWhenReturnTrue) {
    if (n->local_search(sol)) {
        Eigen::VectorXd saved = sol.coverage;
        n->coverage.reset(sol);
        EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9));
    }
}

// On TRUE the coverage constraint must remain satisfied.
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchCoverageConstraintWhenReturnTrue) {
    if (n->local_search(sol)) {
        n->coverage.reset(sol);
        EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
    }
}

// Repeated calls must converge; at the local optimum geometry.cost and the
// stored cost must agree, and the coverage constraint must hold.
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchConvergesToLocalOptimum) {
    static constexpr int max_iters = 20;
    for (int i = 0; i < max_iters; ++i) {
        bool improved = n->local_search(sol);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
        if (!improved) break;
    }
    EXPECT_EQ(n->geometry.cost(sol), sol.cost);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

// After a non-improving call the path must still be completable without
// exception, and the cost after completion must be consistent.
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchPathRecoverableAfterNoImprovement) {
    static constexpr int max_iters = 10;
    bool last = true;
    for (int i = 0; i < max_iters && last; ++i) {
        last = n->local_search(sol);
        if (last) {
            n->coverage.reset(sol);
            sol.cost = n->geometry.cost(sol);
        }
    }
    ASSERT_GT(sol.path.size(), 0u);
    EXPECT_NO_THROW({
        n->geometry.complete(sol);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    });
    EXPECT_EQ(n->geometry.cost(sol), sol.cost);
}

//Improving call must return true, no throw, a valid initial position and coverage update.
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchDoesNotThrowOnImprovement){
    sol.path = {{6, 11},
                  {79, path_engine::NULL_NODE},
                  {23,18},
                  {44,42},
                  {48, 51},
                  {30, 35},
                  {75,74},
                  {10, 9},
                  {62, 63}};
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);
    EXPECT_NO_THROW(n->local_search(sol));
}

TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchFindImprovement){
    sol.path = {{6, 11},
                {79, path_engine::NULL_NODE},
                {23,18},
                {44,42},
                {48, 51},
                {30, 35},
                {75,74},
                {10, 9},
                {62, 63}};
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);
    cost_t c = sol.cost;
    if (n->local_search(sol)){
        EXPECT_LT(n->geometry.cost(sol), c);
    } else {
        EXPECT_EQ(n->geometry.cost(sol), c);
    }
}

// ──── {10,9} replacement tests — same instance as E2ELocalSearchFindImprovement ────
//
// Logical chain being validated:
//   1. add_replacements generates {16,15} as a candidate for {10,9}
//   2. gain({10,9}) > {0,0} — the neighbourhood considers it worth replacing
//   3. local_search produces a path with {16,15}/{15,16} in place of {10,9}
//
// {10,9} is the segment at index 7 in the path above.

// add_replacements must include {16,15} or {15,16} among the alternatives for
// {10,9}.  This is deterministic — it depends only on geometry and coverage,
// not on LKH.
TEST_F(neighborhood_gtsp_fixture, ReplacementsForSegment10_9IncludesNeighboringRow) {
    sol.path = {{6, 11},
                {79, path_engine::NULL_NODE},
                {23, 18},
                {44, 42},
                {48, 51},
                {30, 35},
                {75, 74},
                {10, 9},
                {62, 63}};
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);

    // Build the candidate list starting from {10,9} at index 7.
    std::vector<segment> candidates = {sol.path[7]};
    n->add_replacements(sol, candidates);

    bool found = std::any_of(candidates.begin(), candidates.end(),
        [](const segment& s) {
            return (s.source == 16 && s.target == path_engine::NULL_NODE) ||
                   (s.source == 16 && s.target == path_engine::REVERSED_NULL_NODE);
        });
    EXPECT_TRUE(found)
        << "add_replacements must include {16,15} or {15,16} as a replacement for {10,9}";
}

// The gain computed for {10,9} must be strictly positive.
// gain = (removal_gain * (M>0)) + {0,1} * (M>0); if replacements exist (M>0)
// the gain is at least {0,1} > {0,0}.
TEST_F(neighborhood_gtsp_fixture, GainForSegment10_9IsPositive) {
    sol.path = {{6, 11},
                {79, path_engine::NULL_NODE},
                {23, 18},
                {44, 42},
                {48, 51},
                {30, 35},
                {75, 74},
                {10, 9},
                {62, 63}};
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);

    std::vector<segment> candidates = {sol.path[7]};
    n->add_replacements(sol, candidates);
    int M = static_cast<int>(candidates.size() - 1);

    auto iter = std::next(sol.path.cbegin(), 7);
    cost_t g = n->gain(sol, iter, M);

    EXPECT_GT(g, (cost_t{0, 0}))
        << "gain({10,9}) must be positive (M=" << M
        << "); cluster_selection will not select the segment otherwise";
}

// local_search must produce a path where {10,9} has been replaced by {16,15}
// or {15,16}.  The gain is positive and the replacement is available, so the
// neighbourhood is expected to find the improvement reliably.
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchFindImprovement_Replaces10_9) {
    sol.path = {{6, 11},
                {79, path_engine::NULL_NODE},
                {23, 18},
                {44, 42},
                {48, 51},
                {30, 35},
                {75, 74},
                {10, 9},
                {62, 63}};
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);

    int n_trials = 100;
    int count_improved=0;
    int count_n109_moved=0;
    int count_has_replacement_16 = 0;
    auto sol_tmp = sol;
    for (int i = 0; i< n_trials; i++){
        bool improved = n->local_search(sol);
        count_improved += improved;
        if (improved){
            bool still_has_10_9 = std::any_of(sol.path.begin(), sol.path.end(),
                                              [](const segment& s){ return s.source == 10 && s.target == 9; });
            count_n109_moved += 1 - still_has_10_9;

            bool has_replacement = std::any_of(sol.path.begin(), sol.path.end(),
                                               [](const segment& s) {
                                                   return (s.source == 16 && s.target == path_engine::NULL_NODE) ||
                                                          (s.source == 16 && s.target == path_engine::REVERSED_NULL_NODE);
                                               });
            count_has_replacement_16 += has_replacement;
        }
        sol = sol_tmp;
    }
    EXPECT_GT(count_improved, 0)
        << "local_search must find an improvement for this instance for at least one run.";

    EXPECT_GT(count_n109_moved, 0)
        << "Segment {10,9} must have been replaced by a local_search";

    EXPECT_GT(count_has_replacement_16, 0)
        << "Expected replacement {16,-1} or {16, -2} for {10,9} not found in path after any local_search";
}

TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchAccurateCostOnImprovement){
    sol.path = {{6, 11},
                {79, path_engine::NULL_NODE},
                {23,18},
                {44,42},
                {48, 51},
                {30, 35},
                {75,74},
                {10, 9},
                {62, 63}};
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);
    n->local_search(sol);
    cost_t c = n->geometry.cost(sol);
    EXPECT_EQ(c, sol.cost);
}

//---------Testing specific edge cases---------------------------
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchDoesNotDegradeFirstCase){
    sol.path = {{6, -1},
        {49, -1},
        {12, -2},
        {18, -1},
        {19, -1},
        {20, 21},
        {22, -1},
        {23, -1},
        {11, -2},
        {8, -2},
        {7, -2},
        {6, -2},
        {42, -2},
        {48, -1},
        {50, -1},
        {51, -1},
        {24, -2},
        {30, -1},
        {31, -1},
        {32, 33},
        {34, 35},
        {29, -2},
        {75, -2},
        {74, 73},
        {10, 9},
        {61, -1},
        {62, 63}};
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);
    cost_t c = sol.cost;
    n->local_search(sol);
    EXPECT_EQ(sol.cost, n->geometry.cost(sol));
    EXPECT_GE(c, sol.cost);
}

//TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchFindFeasiblePath){
//    sol.path = {{6, -1},
//            {49, -1},
//            {50, 51},
//            {46, -1},
//            {30, 31},
//            {32, -1},
//            {33, 35},
//            {23, -2},
//            {22, 19},
//            {48, -2}};
//    n->coverage.reset(sol);
//    sol.cost = n->geometry.cost(sol);
//    cost_t c = sol.cost;
//    ASSERT_NO_THROW(n->local_search(sol));
//    EXPECT_EQ(sol.cost, n->geometry.cost(sol));
//    EXPECT_GE(c, sol.cost);
//}

// ============================================================================
// SIGSEGV guards — exit code -11 in k_avg_performance.py
//
// Primary cause: lkh() does not check std::system() return code. When LKH
// fails the stale .sol from a prior (larger) graph is read, producing tour
// indices beyond graph.nodes.size() → OOB access → SIGSEGV.
//
// Secondary cause: sgtsp_to_atsp() sums non-penalty matrix entries with int
// arithmetic; for large graphs this overflows, corrupting the cost matrix and
// causing LKH to produce an invalid tour.
// ============================================================================

// tsp_to_path must throw std::out_of_range when a tour index exceeds the
// graph node count — the second line of defence once a stale .sol is read.
TEST_F(neighborhood_gtsp_fixture, TspToPathThrowsWhenTourIndexExceedsGraphNodes) {
    cppied_solution two;
    two.path     = {{6, 11}, {30, 35}};
    two.coverage = Eigen::VectorXd::Zero(P.req.size());
    n->coverage.reset(two);
    two.cost = n->geometry.cost(two);

    auto g = make_graph(two);  // 2 segments × 3 nodes + 3 dummy = 9 nodes

    // Tour from a stale .sol written for a 10-node graph: valid indices are
    // [1,9] for the current 9-node graph; index 10 is out of range.
    std::vector<int> stale_tour = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    cppied_solution trial = two;
    EXPECT_THROW(n->tsp_to_path(trial, stale_tour, g), std::out_of_range)
        << "tsp_to_path must throw on an out-of-range tour index; "
           "without bounds checking a stale .sol file causes SIGSEGV (-11)";
}

// lkh() must return warm_start when the LKH process fails (system() != 0),
// rather than reading the stale .sol left over from a previous larger graph.
TEST_F(neighborhood_gtsp_e2e_fixture, LkhReturnsFallbackWhenLkhProcessFails) {
    // Write a stale .sol with 999 nodes (far larger than any graph in this test)
    std::filesystem::path sol_path =
        std::filesystem::path(scratch_dir) / (ctx.config.at("NAME") + ".sol");
    {
        std::ofstream f(sol_path);
        f << "NAME : stale\nTYPE : TOUR\nDIMENSION : 999\nTOUR_SECTION\n";
        for (int i = 1; i <= 999; ++i) f << i << "\n";
        f << "-1\n";
    }

    // Replace the LKH binary with a command that always exits non-zero.
    ctx.config["LKH_EXECUTABLE"] = "/usr/bin/false";

    Eigen::MatrixXi C(9, 9);
    C.setConstant(1000);
    std::vector<int> warm_start = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    std::vector<int> result = n->tspEngine.lkh(C, warm_start);

    // Without the fix lkh() ignores the non-zero return code and reads the stale
    // 999-node file, returning a 999-element tour instead of warm_start.
    EXPECT_EQ(result, warm_start)
        << "lkh() must return warm_start when system() fails; "
           "reading a stale .sol with larger DIMENSION causes OOB graph.nodes access";
}
