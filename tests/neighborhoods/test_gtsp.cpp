#include "../test_fixture.hpp"
#include "../../include/neighborhoods/neighborhood.hpp"
#include "../../include/engines/tsp_engine.hpp"
#include "../../include/neighborhoods/neighborhood_trim.hpp"
#include "../../include/neighborhoods/neighborhood_cut.hpp"
// neighborhood_gtsp::trial is private; expose it to the accessor so the
// tour -> path output can be inspected. Dependencies are included above so
// only neighborhood_gtsp itself is affected.
#define private protected
#include "../../include/neighborhoods/neighborhood_gtsp.hpp"
#undef private

struct neighborhood_gtsp_accessor : public neighborhood_gtsp {
public:
    // trial is only copied from the bound solution in local_search(); seed it
    // here so the helpers can be exercised on their own.
    neighborhood_gtsp_accessor(cppied_context& c, cppied_instance& i, cppied_solution& s)
        : neighborhood_gtsp(c, i, s) { trial = s; }

    using neighborhood::geometry;
    using neighborhood::coverage;
    using neighborhood::problem;

    using neighborhood_gtsp::tspEngine;
    using neighborhood_gtsp::trial;
    using neighborhood_gtsp::setReplacements;
    using neighborhood_gtsp::replacements;
    using neighborhood_gtsp::buildGtspInstance;
    using neighborhood_gtsp::getReplacementBucket;
    using neighborhood_gtsp::transformTspTourToSolution;
    using neighborhood_gtsp::getClusterGroupSize;
    using neighborhood_gtsp::selectClusters;
    using neighborhood_gtsp::getUnsatisfiedCells;
};

class neighborhood_gtsp_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<neighborhood_gtsp_accessor> n;
    cppied_solution sol;

    // Neighborhoods are bound to a solution at construction.
    bool search(cppied_solution& s){
        return neighborhood_gtsp_accessor(ctx, P, s).local_search();
    }

    static constexpr const char* scratch_dir =
            PROJECT_SOURCE_DIR "/tests/scratch";

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
        n = std::make_unique<neighborhood_gtsp_accessor>(ctx, P, sol);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }

    // Build a minimal, ready-to-use GTSPGraph from s (no replacements).
    // Non-edges of the cost matrix are set to s.cost.length; g.tour holds the
    // warm start.
    tsp::GTSPGraph make_graph(cppied_solution& s) {
        neighborhood_gtsp_accessor m(ctx, P, s);
        std::vector<int> sel;
        return m.buildGtspInstance(sel);
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

// Build the graph of a one-segment path and transform its warm-start tour
// back to a path: the segment must be recovered.
static segment roundtrip(neighborhood_gtsp_accessor& m) {
    std::vector<int> sel;
    tsp::GTSPGraph g = m.buildGtspInstance(sel);
    m.transformTspTourToSolution(g);
    EXPECT_EQ(m.trial.path.size(), 1u);
    return m.trial.path.empty() ? segment{} : m.trial.path.front();
}

TEST_F(neighborhood_gtsp_fixture, NodesToSegmentRoundtripForward) {
    segment original = {6, 11};
    cppied_solution one;
    one.path     = {original};
    one.coverage = Eigen::VectorXd::Zero(P.req.size());
    n->coverage.reset(one);
    one.cost = n->geometry.cost(one);
    neighborhood_gtsp_accessor m(ctx, P, one);
    segment recovered = roundtrip(m);
    EXPECT_EQ(recovered.source, original.source);
    EXPECT_EQ(recovered.target, original.target);
}

// Round-trip must also work for a reversed (W/N-direction) segment.
TEST_F(neighborhood_gtsp_fixture, NodesToSegmentRoundtripReversed) {
    segment original = {23, 18};   // W direction: source > target, both E-range
    cppied_solution one;
    one.path     = {original};
    one.coverage = Eigen::VectorXd::Zero(P.req.size());
    n->coverage.reset(one);
    one.cost = n->geometry.cost(one);
    neighborhood_gtsp_accessor m(ctx, P, one);
    segment recovered = roundtrip(m);
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
    auto g = make_graph(sol);
    EXPECT_EQ(g.tour.size(), g.nodes.size());
}

// CORRECTNESS CONTRACT: node IDs in graph.clusters must be 0-based so they
// can be used directly as matrix row/column indices in gtsp_to_sgtsp.
// Bug: build_clusters starts node_id at 1, making all cluster.nodes entries
// 1-based — off by one when used as Eigen matrix indices.
TEST_F(neighborhood_gtsp_fixture, BuildClustersNodeIDsAreZeroBased) {
    auto g = make_graph(sol);
    EXPECT_EQ(g.clusters[0].nodes[0], 0)
        << "cluster.nodes must hold 0-based matrix positions; "
           "currently node_id starts at 1 causing off-by-one in gtsp_to_sgtsp";
}

// Each node in graph.nodes at position i must carry id == i so that 1-based
// TSP tour indices (tour[k]-1) map back correctly to graph.nodes positions.
TEST_F(neighborhood_gtsp_fixture, BuildClustersNodeIDMatchesPosition) {
    auto g = make_graph(sol);
    for (int i = 0; i < static_cast<int>(g.nodes.size()); ++i)
        EXPECT_EQ(g.nodes[i].id, i)
            << "node at position " << i << " has id " << g.nodes[i].id
            << "; id must equal position for correct matrix/tour indexing";
}

// ============================================================================
// set_gtsp_costs — cost matrix content
// ============================================================================

// Root block (positions 0..5): intra edges are zero.
TEST_F(neighborhood_gtsp_fixture, SetGtspCostsRootIntraEdgesAreZero) {
    auto g = make_graph(sol);
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(g.cost(i, i+1), 0) << "root intra edge (" << i << "," << i+1 << ")";
}

// Every other segment triple (positions i, i+1, i+2 with i = 6, 9, ...):
// the segment length is split across its two intra edges as
// floor(length/2) (i→i+1) and ceil(length/2) (i+1→i+2).
TEST_F(neighborhood_gtsp_fixture, SetGtspCostsIntraSegmentEdgesSplitLength) {
    auto g = make_graph(sol);
    for (int i = 6; i < static_cast<int>(g.nodes.size()) - 2; i += 3) {
        segment s = {n->geometry.flip_segment(g.nodes[i].s).source,
                     n->geometry.flip_segment(g.nodes[i+2].s).source};
        if (s.source == s.target) s.target = path_engine::NULL_NODE;
        int length = n->geometry.cost(s).length;
        EXPECT_EQ(g.cost(i,   i+1), length / 2)
            << "intra edge (" << i << "," << i+1 << "), length " << length;
        EXPECT_EQ(g.cost(i+1, i+2), (length + 1) / 2)
            << "intra edge (" << i+1 << "," << i+2 << "), length " << length;
    }
}

// The two intra edges of a segment sum to its length and differ by at most 1.
TEST_F(neighborhood_gtsp_fixture, SetGtspCostsIntraSegmentEdgesBalanced) {
    auto g = make_graph(sol);
    for (int i = 6; i < static_cast<int>(g.nodes.size()) - 2; i += 3) {
        int a = g.cost(i, i+1), b = g.cost(i+1, i+2);
        EXPECT_GE(a, 0);
        EXPECT_TRUE(b == a || b == a + 1)
            << "intra edges (" << i << ") not floor/ceil halves: " << a << ", " << b;
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
// must be set to an actual geometry distance, not left at the non-edge
// sentinel (trial cost length).
// Positions 0..2 are the dummy chain (0-1-2-3 only) and are skipped; segment
// triples start at position 3.
TEST_F(neighborhood_gtsp_fixture, SetGtspCostsCrossSegmentEdgesAreSet) {
    auto g = make_graph(sol);
    const int sentinel = sol.cost.length;
    for (int i = 3; i < static_cast<int>(g.nodes.size()) - 5; i += 3) {
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
    EXPECT_NO_THROW(search(sol));
}

// The path must be non-empty after the call regardless of return value.
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchPathNonEmpty) {
    search(sol);
    EXPECT_GT(sol.path.size(), 0u);
}

// On TRUE the tracked cost must be strictly less than the pre-call incumbent.
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchCostDecreasedWhenReturnTrue) {
    cost_t before = sol.cost;
    if (search(sol))
        EXPECT_LT(sol.cost, before);
}

// On TRUE the tracked cost must equal geometry.cost(sol) (trim+cut leave the
// cost field consistent).
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchCostConsistentWhenReturnTrue) {
    if (search(sol))
        EXPECT_EQ(n->geometry.cost(sol), sol.cost);
}

// On TRUE the coverage vector must match a fresh reset (incremental updates
// must agree with a full recompute).
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchCoverageConsistentWhenReturnTrue) {
    if (search(sol)) {
        Eigen::VectorXd saved = sol.coverage;
        n->coverage.reset(sol);
        EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9));
    }
}

// On TRUE the coverage constraint must remain satisfied.
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchCoverageConstraintWhenReturnTrue) {
    if (search(sol)) {
        n->coverage.reset(sol);
        EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
    }
}

// Repeated calls must converge; at the local optimum geometry.cost and the
// stored cost must agree, and the coverage constraint must hold.
TEST_F(neighborhood_gtsp_e2e_fixture, E2ELocalSearchConvergesToLocalOptimum) {
    static constexpr int max_iters = 20;
    for (int i = 0; i < max_iters; ++i) {
        bool improved = search(sol);
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
        last = search(sol);
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
    EXPECT_NO_THROW(search(sol));
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
    if (search(sol)){
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

    // Replacement bucket of {10,9} at index 7.
    neighborhood_gtsp_accessor m(ctx, P, sol);
    std::vector<segment> candidates = m.getReplacementBucket(7).replacements;

    bool found = std::any_of(candidates.begin(), candidates.end(),
        [](const segment& s) {
            return (s.source == 16 && s.target == path_engine::NULL_NODE) ||
                   (s.source == 16 && s.target == path_engine::REVERSED_NULL_NODE);
        });
    EXPECT_TRUE(found)
        << "getReplacementBucket must include {16,15} or {15,16} as a replacement for {10,9}";
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

    neighborhood_gtsp_accessor m(ctx, P, sol);
    ReplacementBucket bucket = m.getReplacementBucket(7);
    int M = bucket.size();

    auto iter = std::next(sol.path.cbegin(), 7);
    cost_t g = m.gain(sol, iter, M);

    EXPECT_GT(g, (cost_t{0, 0}))
        << "gain({10,9}) must be positive (M=" << M
        << "); selectClusters will not select the segment otherwise";
    EXPECT_EQ(bucket.gain, g)
        << "bucket gain must match gain() for the same replacement count";
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
        bool improved = search(sol);
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
    search(sol);
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
    search(sol);
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
//    ASSERT_NO_THROW(search(sol));
//    EXPECT_EQ(sol.cost, n->geometry.cost(sol));
//    EXPECT_GE(c, sol.cost);
//}

// ============================================================================
// transformTspTourToSolution — valid tours only
//
// Precondition: every tour index lies in [1, graph.nodes.size()] and each
// cluster group is traversed contiguously (what lkh returns for a graph built
// by buildGtspInstance). Tours below are derived from the graph's own warm
// start, so they satisfy the precondition by construction.
// ============================================================================

// Rewrite the tour span of the cluster group of path index `group` (starting
// at tour position `pos`) so that alternative `alt` (0 = original segment) is
// traversed first, entry-first or exit-first. Each cluster's nodes stay
// contiguous, as in an LKH tour.
static void setGroupTraversal(std::vector<int>& tour, const tsp::GTSPGraph& g,
                              int pos, int group, int alt, bool exitFirst) {
    std::array<int, 3> order = exitFirst ? std::array<int, 3>{2, 1, 0}
                                         : std::array<int, 3>{0, 1, 2};
    for (int c : order) {
        std::vector<int> nodes = g.clusters[3 + 3 * group + c].nodes;
        std::rotate(nodes.begin(), std::next(nodes.begin(), alt), nodes.end());
        for (int node : nodes) tour[pos++] = node + 1;
    }
}

// Reverse the traversal of a cluster group: exit cluster, middle, entry.
static void reverseGroupTraversal(std::vector<int>& tour, int pos, int clusterSize) {
    auto first = std::next(tour.begin(), pos);
    auto last  = std::next(first, 3 * clusterSize);
    std::reverse(first, last);
}

static void expectConsistentTrial(neighborhood_gtsp_accessor& m, cppied_instance& P) {
    cppied_solution fresh = m.trial;
    m.coverage.reset(fresh);
    EXPECT_TRUE(fresh.coverage.isApprox(m.trial.coverage)) << "trial coverage not reset";
    EXPECT_EQ(m.trial.cost, m.geometry.cost(m.trial)) << "trial cost not recomputed";
    (void)P;
}

// The warm-start tour maps back to the original path, cost and coverage.
TEST_F(neighborhood_gtsp_fixture, TransformWarmStartTourRecoversPath) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    std::vector<int> sel;
    auto g = m.buildGtspInstance(sel);
    m.transformTspTourToSolution(g);

    ASSERT_EQ(m.trial.path.size(), sol.path.size());
    for (size_t i = 0; i < sol.path.size(); ++i)
        EXPECT_EQ(m.trial.path[i], sol.path[i]) << "segment " << i;
    EXPECT_EQ(m.trial.cost, sol.cost);
    expectConsistentTrial(m, P);
}

// LKH may return the tour in either direction (written starting at node 1):
// 1, N, N-1, ..., 2 must give the same path.
TEST_F(neighborhood_gtsp_fixture, TransformReversedWarmStartTourRecoversPath) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    std::vector<int> sel;
    auto g = m.buildGtspInstance(sel);
    std::reverse(std::next(g.tour.begin()), g.tour.end());
    m.transformTspTourToSolution(g);

    ASSERT_EQ(m.trial.path.size(), sol.path.size());
    for (size_t i = 0; i < sol.path.size(); ++i)
        EXPECT_EQ(m.trial.path[i], sol.path[i]) << "segment " << i;
    expectConsistentTrial(m, P);
}

// Same cycle as above written with node 1 last: N, ..., 2, 1.
TEST_F(neighborhood_gtsp_fixture, TransformReversedTourEndingAtRootRecoversPath) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    std::vector<int> sel;
    auto g = m.buildGtspInstance(sel);
    std::reverse(g.tour.begin(), g.tour.end());
    m.transformTspTourToSolution(g);

    ASSERT_EQ(m.trial.path.size(), sol.path.size());
    for (size_t i = 0; i < sol.path.size(); ++i)
        EXPECT_EQ(m.trial.path[i], sol.path[i]) << "segment " << i;
}

// A rotated warm start (starting anywhere in the cycle) must give the same path.
TEST_F(neighborhood_gtsp_fixture, TransformRotatedWarmStartTourRecoversPath) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    std::vector<int> sel;
    auto g = m.buildGtspInstance(sel);
    std::rotate(g.tour.begin(), std::next(g.tour.begin(), 10), g.tour.end());
    m.transformTspTourToSolution(g);

    ASSERT_EQ(m.trial.path.size(), sol.path.size());
    for (size_t i = 0; i < sol.path.size(); ++i)
        EXPECT_EQ(m.trial.path[i], sol.path[i]) << "segment " << i;
}

// Traversing a group exit-first yields the flipped segment at that position,
// every other segment unchanged.
TEST_F(neighborhood_gtsp_fixture, TransformExitFirstGroupFlipsSegment) {
    const int k = 4;                                  // {48,51}
    neighborhood_gtsp_accessor m(ctx, P, sol);
    std::vector<int> sel;
    auto g = m.buildGtspInstance(sel);
    reverseGroupTraversal(g.tour, 3 + 3 * k, 1);
    m.transformTspTourToSolution(g);

    ASSERT_EQ(m.trial.path.size(), sol.path.size());
    for (int i = 0; i < static_cast<int>(sol.path.size()); ++i) {
        if (i == k)
            EXPECT_EQ(m.trial.path[i], m.geometry.flip_segment(sol.path[i]));
        else
            EXPECT_EQ(m.trial.path[i], sol.path[i]) << "segment " << i;
    }
    expectConsistentTrial(m, P);
}

// Exit-first traversal of a single-node group keeps the node and flips its
// orientation.
TEST_F(neighborhood_gtsp_fixture, TransformExitFirstSingleNodeFlipsOrientation) {
    sol.path = {{6, 11}, {79, path_engine::NULL_NODE}, {23, 18}};
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);
    neighborhood_gtsp_accessor m(ctx, P, sol);
    std::vector<int> sel;
    auto g = m.buildGtspInstance(sel);
    reverseGroupTraversal(g.tour, 3 + 3 * 1, 1);
    m.transformTspTourToSolution(g);

    ASSERT_EQ(m.trial.path.size(), sol.path.size());
    EXPECT_EQ(m.trial.path[1], m.geometry.flip_segment(sol.path[1]));
    EXPECT_EQ(m.trial.path[0], sol.path[0]);
    EXPECT_EQ(m.trial.path[2], sol.path[2]);
}

// Selecting a replacement node in a cluster group puts that replacement in
// place of the original segment (entry-first: as generated; exit-first:
// flipped); the result stays coverage-feasible and its cost/coverage are
// recomputed. On this instance the flipped {16,-2} is the improving move.
class neighborhood_gtsp_replacement_fixture : public neighborhood_gtsp_fixture {
protected:
    static constexpr int k = 7;                       // {10,9}

    void SetUp() override {
        neighborhood_gtsp_fixture::SetUp();
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
    }

    // Transform a tour selecting replacement {16,*} of {10,9}; returns it.
    segment transformSelecting16(neighborhood_gtsp_accessor& m, bool exitFirst) {
        m.setReplacements();
        ReplacementBucket bucket = m.getReplacementBucket(k);
        auto it = std::find_if(bucket.replacements.begin(), bucket.replacements.end(),
                               [](const segment& s){ return s.source == 16; });
        EXPECT_NE(it, bucket.replacements.end()) << "{16,*} must be a replacement for {10,9}";
        if (it == bucket.replacements.end()) return {};
        const int alt = 1 + static_cast<int>(std::distance(bucket.replacements.begin(), it));

        std::vector<int> sel = {k};
        auto g = m.buildGtspInstance(sel);
        // Groups before k hold one alternative each.
        setGroupTraversal(g.tour, g, 3 + 3 * k, k, alt, exitFirst);
        m.transformTspTourToSolution(g);
        return *it;
    }

    void expectOnlyGroupReplaced(neighborhood_gtsp_accessor& m, const segment& expected) {
        ASSERT_EQ(m.trial.path.size(), sol.path.size());
        for (int i = 0; i < static_cast<int>(sol.path.size()); ++i) {
            if (i == k)
                EXPECT_EQ(m.trial.path[i], expected);
            else
                EXPECT_EQ(m.trial.path[i], sol.path[i]) << "segment " << i;
        }
        EXPECT_TRUE((m.trial.coverage.array() >= P.req.array()).all());
        expectConsistentTrial(m, P);
    }
};

TEST_F(neighborhood_gtsp_replacement_fixture, TransformEntryFirstSelectsReplacement) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    segment r = transformSelecting16(m, false);
    expectOnlyGroupReplaced(m, r);
}

TEST_F(neighborhood_gtsp_replacement_fixture, TransformExitFirstSelectsFlippedReplacement) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    segment r = transformSelecting16(m, true);
    expectOnlyGroupReplaced(m, m.geometry.flip_segment(r));
    EXPECT_LT(m.trial.cost, sol.cost) << "{16,-2} in place of {10,9} is an improvement";
}

// The original alternative of a group with replacements maps back to the
// original segment.
TEST_F(neighborhood_gtsp_replacement_fixture, TransformOriginalAlternativeKeepsSegment) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    m.setReplacements();
    std::vector<int> sel = {k};
    auto g = m.buildGtspInstance(sel);
    m.transformTspTourToSolution(g);
    expectOnlyGroupReplaced(m, sol.path[k]);
    EXPECT_EQ(m.trial.cost, sol.cost);
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

// ============================================================================
// Replacement buckets
// ============================================================================

class neighborhood_gtsp_bucket_fixture : public neighborhood_gtsp_fixture {
protected:
    void SetUp() override {
        neighborhood_gtsp_fixture::SetUp();
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
    }
};

// Range-for over a bucket visits each replacement exactly once.
// Regression: end() returned begin(), so range-for loops saw no replacement.
TEST_F(neighborhood_gtsp_bucket_fixture, BucketRangeForVisitsEveryReplacement) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    ReplacementBucket bucket = m.getReplacementBucket(7);
    ASSERT_GT(bucket.size(), 0);
    int visited = 0;
    for (auto& s : bucket) { (void)s; ++visited; }
    EXPECT_EQ(visited, bucket.size());
}

// A selected group holds the original plus every replacement of its bucket.
TEST_F(neighborhood_gtsp_bucket_fixture, SelectedGroupContainsAllReplacements) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    m.setReplacements();
    std::vector<int> sel = {7};
    auto g = m.buildGtspInstance(sel);
    const int M = m.replacements[7].size();
    ASSERT_GT(M, 0);
    EXPECT_EQ(g.nodes.size(), 3 + 3 * (sol.path.size() + M));
    for (int c = 0; c < 3; ++c)
        EXPECT_EQ(g.clusters[3 + 3 * 7 + c].nodes.size(), static_cast<size_t>(1 + M));
}

// Buckets never contain the original segment or its reversal (both are
// already represented by the original's entry/exit nodes).
TEST_F(neighborhood_gtsp_bucket_fixture, BucketExcludesSegmentAndItsReversal) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    for (int i = 1; i < static_cast<int>(sol.path.size()); ++i) {
        ReplacementBucket bucket = m.getReplacementBucket(i);
        segment reversed = m.geometry.flip_segment(sol.path[i]);
        for (auto& s : bucket.replacements) {
            EXPECT_NE(s, sol.path[i]) << "bucket " << i << " contains its own segment";
            EXPECT_NE(s, reversed) << "bucket " << i << " contains its own reversal";
        }
        if (bucket.size() == 0)
            EXPECT_EQ(bucket.gain, (cost_t{0, 0}))
                << "bucket " << i << " has a gain without any real replacement";
    }
}

// setReplacements starts from scratch on every call (one bucket per segment).
TEST_F(neighborhood_gtsp_bucket_fixture, SetReplacementsDoesNotAccumulate) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    m.setReplacements();
    m.setReplacements();
    EXPECT_EQ(m.replacements.size(), sol.path.size());
}

// ============================================================================
// GTSP -> ATSP transformation (Noon-Bean)
//
// A clustered tour uses exactly K = #clusters arcs leaving a cluster; any
// other tour uses at least K+1. The transformation must make every clustered
// tour cheaper than every non-clustered one: (K+1)*min > K*max over the arcs
// that are not intra-cluster cycle arcs. Regression: sgtsp_to_atsp was the
// identity, LKH returned interleaved clusters and the tour parse produced
// malformed (shorter, coverage-infeasible) paths.
// ============================================================================

static bool isIntraClusterCycleArc(const tsp::GTSPGraph& g, const std::vector<int>& clusterOf, int u, int v) {
    if (clusterOf[u] != clusterOf[v]) return false;
    const auto& nodes = g.clusters[clusterOf[u]].nodes;
    if (nodes.size() < 2) return false;
    for (size_t k = 0; k < nodes.size(); ++k)
        if (nodes[k] == u && nodes[(k + 1) % nodes.size()] == v) return true;
    return false;
}

TEST_F(neighborhood_gtsp_bucket_fixture, AtspTransformForcesClusteredTours) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    m.setReplacements();
    std::vector<int> sel = {1, 3, 7};
    auto g = m.buildGtspInstance(sel);
    ASSERT_GT(g.clusters[3 + 3 * 7].nodes.size(), 1u) << "need multi-node clusters";
    m.tspEngine.gtsp_to_atsp(g);

    const int n_nodes = static_cast<int>(g.nodes.size());
    std::vector<int> clusterOf(n_nodes, -1);
    for (int c = 0; c < static_cast<int>(g.clusters.size()); ++c)
        for (int v : g.clusters[c].nodes) clusterOf[v] = c;

    long long lo = std::numeric_limits<long long>::max(), hi = 0;
    for (int u = 0; u < n_nodes; ++u)
        for (int v = 0; v < n_nodes; ++v) {
            if (u == v) continue;
            if (isIntraClusterCycleArc(g, clusterOf, u, v)) {
                EXPECT_EQ(g.cost(u, v), 0) << "cycle arc (" << u << "," << v << ")";
                continue;
            }
            lo = std::min<long long>(lo, g.cost(u, v));
            hi = std::max<long long>(hi, g.cost(u, v));
        }
    const long long K = static_cast<long long>(g.clusters.size());
    EXPECT_GT((K + 1) * lo, K * hi)
        << "a non-clustered tour can be cheaper than a clustered one (min " << lo << ", max " << hi << ")";
    EXPECT_LT(K * hi, static_cast<long long>(std::numeric_limits<int>::max()))
        << "tour costs may overflow int";
}

// ============================================================================
// trial is taken from the bound solution at each local_search call
// ============================================================================

// The neighborhood is bound (and its trial seeded) before the solution
// changes, as in VND where earlier neighborhoods modify pSol. The current
// solution is the bound one plus an extra segment, so a result built from the
// stale trial would be cheaper and accepted. GTSP keeps one segment per
// group, so a result built from the current solution keeps its size.
// Regression: trial was copied at construction (stale).
TEST_F(neighborhood_gtsp_e2e_fixture, LocalSearchUsesCurrentSolution) {
    neighborhood_gtsp_accessor m(ctx, P, sol);   // trial = 7-segment fixture path
    sol.path.push_back({0, 1});
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);
    const size_t size = sol.path.size();
    m.local_search();
    EXPECT_EQ(sol.path.size(), size);
    cppied_solution fresh = sol;
    n->coverage.reset(fresh);
    EXPECT_TRUE((fresh.coverage.array() >= P.req.array()).all());
}

// ============================================================================
// Unit tests of the GTSP construction helpers
// ============================================================================

// The group of path[k] spans 3*(1+M) tour positions (entry, middle and exit
// clusters of 1+M alternatives), whichever of its entry or exit nodes the
// tour reaches first.
TEST_F(neighborhood_gtsp_bucket_fixture, ClusterGroupSizeIsThreeTimesAlternatives) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    m.setReplacements();
    std::vector<int> sel = {1, 3, 7};
    auto g = m.buildGtspInstance(sel);
    for (int k = 0; k < static_cast<int>(sol.path.size()); ++k) {
        const int alternatives = 1 + (std::count(sel.begin(), sel.end(), k) ? m.replacements[k].size() : 0);
        for (int role : {0, 2}) {
            for (int node : g.clusters[3 + 3 * k + role].nodes)
                EXPECT_EQ(m.getClusterGroupSize(g, node + 1), 3 * alternatives)
                    << "group " << k << " role " << role << " node " << node;
        }
    }
}

// The warm-start tour visits every node once.
TEST_F(neighborhood_gtsp_bucket_fixture, BuildGtspInstanceTourIsPermutation) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    m.setReplacements();
    std::vector<int> sel = {1, 3, 7};
    auto g = m.buildGtspInstance(sel);
    std::vector<int> tour = g.tour, expected(g.nodes.size());
    std::sort(tour.begin(), tour.end());
    std::iota(expected.begin(), expected.end(), 1);
    EXPECT_EQ(tour, expected);
    EXPECT_EQ(g.cost.rows(), static_cast<int>(g.nodes.size()));
    EXPECT_TRUE(g.cost.isApprox(g.cost.transpose())) << "GTSP cost matrix must be symmetric";
}

// selectClusters only selects groups with a positive bucket gain, never the
// start segment, and groups with pairwise disjoint footprints.
TEST_F(neighborhood_gtsp_bucket_fixture, SelectClustersInvariants) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    m.setReplacements();
    std::set<int> seen;
    for (int draw = 0; draw < 50; ++draw) {
        auto sel = m.selectClusters();
        for (int i : sel) {
            seen.insert(i);
            EXPECT_GT(i, 0) << "start segment selected";
            EXPECT_GT(m.replacements[i].gain, (cost_t{0, 0})) << "bucket " << i;
        }
        for (size_t a = 0; a < sel.size(); ++a)
            for (size_t b = a + 1; b < sel.size(); ++b) {
                Eigen::VectorXi mask = Eigen::VectorXi::Zero(P.req.size());
                m.coverage.mask(sol, sol.path[sel[a]], mask);
                EXPECT_FALSE(m.coverage.intersect_mask(sol, sol.path[sel[b]], mask))
                    << "groups " << sel[a] << " and " << sel[b] << " overlap";
            }
    }
    for (int i = 1; i < static_cast<int>(sol.path.size()); ++i)
        if (m.replacements[i].gain > cost_t{0, 0})
            EXPECT_TRUE(seen.count(i)) << "positive bucket " << i << " never selected in 50 draws";
}

// Every replacement in a bucket restores feasibility when it takes the
// original segment's place.
TEST_F(neighborhood_gtsp_bucket_fixture, BucketReplacementsRestoreCoverage) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    for (int i = 1; i < static_cast<int>(sol.path.size()); ++i) {
        ReplacementBucket bucket = m.getReplacementBucket(i);
        for (const segment& r : bucket.replacements) {
            cppied_solution t = sol;
            t.path[i] = r;
            m.coverage.reset(t);
            EXPECT_TRUE((t.coverage.array() >= P.req.array()).all())
                << "bucket " << i << " replacement {" << r.source << "," << r.target << "}";
        }
        // The trial is left untouched by the bucket computation.
        cppied_solution fresh = m.trial;
        m.coverage.reset(fresh);
        EXPECT_TRUE(fresh.coverage.isApprox(m.trial.coverage)) << "bucket " << i << " left trial coverage modified";
    }
}

// gain() is zero without replacements and positive otherwise.
TEST_F(neighborhood_gtsp_bucket_fixture, GainIsZeroWithoutReplacements) {
    neighborhood_gtsp_accessor m(ctx, P, sol);
    for (int i = 1; i < static_cast<int>(sol.path.size()); ++i) {
        auto it = std::next(sol.path.cbegin(), i);
        EXPECT_EQ(m.gain(sol, it, 0), (cost_t{0, 0})) << "segment " << i;
        EXPECT_GT(m.gain(sol, it, 1), (cost_t{0, 0})) << "segment " << i;
    }
}

// A redundant segment (removable without unmet cells) has no unsatisfied
// cell, hence an empty bucket and a zero gain.
TEST_F(neighborhood_gtsp_bucket_fixture, RedundantSegmentHasEmptyBucket) {
    sol.path.push_back(sol.path[4]);      // duplicate {48,51}
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);
    neighborhood_gtsp_accessor m(ctx, P, sol);
    const int last = static_cast<int>(sol.path.size()) - 1;
    ReplacementBucket bucket = m.getReplacementBucket(last);
    EXPECT_EQ(bucket.size(), 0);
    EXPECT_EQ(bucket.gain, (cost_t{0, 0}));
}
