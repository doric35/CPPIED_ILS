#include "../test_fixture.hpp"
#include "../../include/construction/dp_sweeper.hpp"

struct dp_sweeper_accessor : public DpSweeper {
public:
    using DpSweeper::DpSweeper;
    using DpSweeper::geometry;
    using DpSweeper::coverage;
    using DpSweeper::problem;
    using DpSweeper::unsatisfiedCells;
    using DpSweeper::setUnsatisfiedCells;
    using DpSweeper::constructCandidateSets;
    using DpSweeper::filterCandidateSet;
    using DpSweeper::filterCandidateSetsWithSpacingConstraint;
    using DpSweeper::selectCandidateSet;
    using DpSweeper::addCoverageSegmentSet;
    using DpSweeper::horizontalDominatesVertical;
    using DpSweeper::getLexicographicValues;
    using DpSweeper::isValidChoice;
    using DpSweeper::addSelectedSegmentsToSolution;
    using DpSweeper::constructArrayCosts;
    using DpSweeper::getArrayBounds;
    using DpSweeper::maximumSubarray;
    using DpSweeper::setRootSegmentIfEmptySolution;
    using DpSweeper::setInitialSolutionAlternateOrientations;
    using DpSweeper::solveTspReduction;
    using DpSweeper::mPenalty;
};

// sol  : fully covering path, bound to dps.
// empty: zero coverage, no path, bound to dpe.
class dp_sweeper_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<dp_sweeper_accessor> dps;
    std::unique_ptr<dp_sweeper_accessor> dpe;
    cppied_solution sol;
    cppied_solution empty;

    void SetUp() override{
        cppied_context_fixture::SetUp();
        sol.path =  {{6, 11},
                     {23,18},
                     {44,42},
                     {48, 51},
                     {30, 35},
                     {75,73},
                     {62, 63}};
        sol.cost = {0,0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        empty.cost = {0,0};
        empty.coverage = Eigen::VectorXd::Zero(P.req.size());
        dps = std::make_unique<dp_sweeper_accessor>(ctx, P, sol);
        dpe = std::make_unique<dp_sweeper_accessor>(ctx, P, empty);
        dps->coverage.reset(sol);
        sol.cost = dps->geometry.cost(sol);
        ctx.start_time = std::chrono::high_resolution_clock::now();
    }

    // Reference marginal gain of vertex v (sum over the cells v observes).
    double expected_vertex_gain(const cppied_solution& s, int v, double penalty){
        double marginal_gain = 0.0;
        for (SMdIt c(P.s_pod, v); c; ++c){
            double local_gain = std::min(P.req(c.index()) - s.coverage(c.index()), c.value());
            marginal_gain += local_gain >= 0 ? local_gain : -penalty;
        }
        return marginal_gain;
    }

    static sweeper::SegmentSet single_vertex_set(const std::vector<int>& sources,
                                                 const std::vector<double>& gains){
        sweeper::SegmentSet S;
        for (int s : sources) S.segments.push_back({s, path_engine::NULL_NODE});
        S.gains = gains;
        return S;
    }
};

TEST_F(dp_sweeper_fixture, NoAllignmentErrorOnInstanciation){
    EXPECT_TRUE(empty.path.empty());
}

// ---------------------------------------------------------------------------
// getArrayBounds / constructArrayCosts
// ---------------------------------------------------------------------------

TEST_F(dp_sweeper_fixture, ArrayBoundsOfForwardSegment){
    auto [u, v] = dpe->getArrayBounds({0, 5});
    EXPECT_EQ(u, 0);
    EXPECT_EQ(v, 5);
}

TEST_F(dp_sweeper_fixture, ArrayBoundsOfReversedSegment){
    auto [u, v] = dpe->getArrayBounds({5, 0});
    EXPECT_EQ(u, 0);
    EXPECT_EQ(v, 5);
}

TEST_F(dp_sweeper_fixture, ArrayBoundsOfSingleVertex){
    auto [u, v] = dpe->getArrayBounds({3, path_engine::NULL_NODE});
    EXPECT_EQ(u, 3);
    EXPECT_EQ(v, 3);
}

TEST_F(dp_sweeper_fixture, ArrayCostsHaveOneEntryPerVertex){
    EXPECT_EQ(dpe->constructArrayCosts({0, 5}).size(), 6u);
    EXPECT_EQ(dpe->constructArrayCosts({47, 42}).size(), 6u);
    EXPECT_EQ(dpe->constructArrayCosts({3, path_engine::NULL_NODE}).size(), 1u);
}

// Entry k of the array costs must be the gain of vertex u+k, i.e. it must
// match the costs of the single-vertex array {u+k}.
TEST_F(dp_sweeper_fixture, ArrayCostsMatchSingleVertexCosts){
    for (segment ref : {segment{0, 5}, segment{42, 47}}) {
        auto costs = dpe->constructArrayCosts(ref);
        for (int k = 0; k < static_cast<int>(costs.size()); ++k) {
            auto single = dpe->constructArrayCosts({ref.source + k, path_engine::NULL_NODE});
            EXPECT_DOUBLE_EQ(costs[k], single[0]) << "array " << ref << " at offset " << k;
        }
    }
}

// The gain of a vertex is the sum of the marginal gains of the cells it observes.
TEST_F(dp_sweeper_fixture, ArrayCostsMatchReferenceGain){
    for (segment ref : {segment{0, 5}, segment{42, 47}}) {
        auto costs = dpe->constructArrayCosts(ref);
        for (int k = 0; k < static_cast<int>(costs.size()); ++k) {
            double expected = expected_vertex_gain(empty, ref.source + k, dpe->get_penalty());
            if (expected > 0.0)
                EXPECT_DOUBLE_EQ(costs[k], expected) << "array " << ref << " at offset " << k;
            else
                EXPECT_LT(costs[k], 0.0) << "array " << ref << " at offset " << k;
        }
    }
}

// On a satisfied row every vertex has non-positive marginal gain.
TEST_F(dp_sweeper_fixture, ArrayCostsNegativeWhenSatisfied){
    for (double c : dps->constructArrayCosts({6, 11}))
        EXPECT_LT(c, 0.0);
}

// ---------------------------------------------------------------------------
// maximumSubarray
// ---------------------------------------------------------------------------

TEST_F(dp_sweeper_fixture, MaximumSubarrayPositiveGainWhenUnsatisfied){
    auto [seg, gain] = dpe->maximumSubarray({0, 5});
    EXPECT_GT(gain, 0.0);
}

TEST_F(dp_sweeper_fixture, MaximumSubarrayResultIsSubsegment){
    auto [seg, gain] = dpe->maximumSubarray({0, 5});
    int lo = path_engine::is_node(seg.target) ? std::min(seg.source, seg.target) : seg.source;
    int hi = path_engine::is_node(seg.target) ? std::max(seg.source, seg.target) : seg.source;
    EXPECT_GE(lo, 0);
    EXPECT_LE(hi, 5);
}

// A single vertex subarray must be encoded as {v, NULL_NODE}, never {v, v}.
TEST_F(dp_sweeper_fixture, MaximumSubarraySingleVertex){
    auto [seg, gain] = dpe->maximumSubarray({3, path_engine::NULL_NODE});
    EXPECT_EQ(seg.source, 3);
    EXPECT_FALSE(path_engine::is_node(seg.target));
}

TEST_F(dp_sweeper_fixture, MaximumSubarrayNeverReturnsDegenerateSegment){
    for (segment ref : {segment{0, 5}, segment{6, 11}, segment{42, 47}, segment{3, path_engine::NULL_NODE}}) {
        auto [seg, gain] = dpe->maximumSubarray(ref);
        EXPECT_NE(seg.source, seg.target) << "from " << ref;
        auto [seg2, gain2] = dps->maximumSubarray(ref);
        EXPECT_NE(seg2.source, seg2.target) << "from " << ref;
    }
}

TEST_F(dp_sweeper_fixture, MaximumSubarrayNonPositiveWhenSatisfied){
    // sol has full coverage reset; row 1 (vertices 6–11) is fully covered.
    auto [seg, gain] = dps->maximumSubarray({6, 11});
    EXPECT_LE(gain, 0.0);
}

TEST_F(dp_sweeper_fixture, MaximumSubarrayVerticalPositiveGain){
    auto [seg, gain] = dpe->maximumSubarray({42, 47});
    EXPECT_GT(gain, 0.0);
    int lo = path_engine::is_node(seg.target) ? std::min(seg.source, seg.target) : seg.source;
    int hi = path_engine::is_node(seg.target) ? std::max(seg.source, seg.target) : seg.source;
    EXPECT_GE(lo, 42);
    EXPECT_LE(hi, 47);
}

// The returned gain is the sum of the array costs over the returned subsegment.
TEST_F(dp_sweeper_fixture, MaximumSubarrayGainMatchesCosts){
    for (segment ref : {segment{0, 5}, segment{47, 42}}) {
        auto costs = dpe->constructArrayCosts(ref);
        auto [seg, gain] = dpe->maximumSubarray(ref);
        auto [u, v] = dpe->getArrayBounds(ref);
        int lo = seg.source;
        int hi = path_engine::is_node(seg.target) ? seg.target : seg.source;
        if (lo > hi) std::swap(lo, hi);
        double sum = 0.0;
        for (int w = lo; w <= hi; ++w) sum += costs[w - u];
        EXPECT_DOUBLE_EQ(gain, sum) << "from " << ref;
    }
}

// ---------------------------------------------------------------------------
// filterCandidateSet (spacing 2*max_range = 2 on the test instance)
// ---------------------------------------------------------------------------

TEST_F(dp_sweeper_fixture, FilterSpacingIsTwiceMaxRange){
    ASSERT_EQ(P.max_range, 1);  // the expectations below assume spacing 2
}

// Candidate sets smaller than the spacing occur near the grid border. Run in a
// subprocess: the check exits with 0 on the expected result, so a crash is
// reported as a failure instead of aborting the test binary.
TEST_F(dp_sweeper_fixture, FilterSingleSegment){
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    auto check = [&]{
        sweeper::SegmentSet S{{{6, 11}}, {5.0}};
        dps->filterCandidateSet(S);
        return S.segments.size() == 1u && S.gains.size() == 1u
               && S.segments[0].source == 6 && S.gains[0] == 5.0;
    };
    EXPECT_EXIT(std::exit(check() ? 0 : 1), ::testing::ExitedWithCode(0), "");
}

TEST_F(dp_sweeper_fixture, FilterSpacingConstraint){
    auto S = single_vertex_set({6, 7}, {3.0, 5.0});
    dps->filterCandidateSet(S);
    ASSERT_EQ(S.segments.size(), 1u);
    EXPECT_EQ(S.segments[0].source, 7);  // higher gain wins
}

TEST_F(dp_sweeper_fixture, FilterRetainsCompatibleSegments){
    auto S = single_vertex_set({6, 7, 8}, {3.0, 1.0, 4.0});
    dps->filterCandidateSet(S);
    ASSERT_EQ(S.segments.size(), 2u);
    EXPECT_EQ(S.segments[0].source, 6);
    EXPECT_EQ(S.segments[1].source, 8);
    EXPECT_EQ(S.gains, (std::vector<double>{3.0, 4.0}));
}

TEST_F(dp_sweeper_fixture, FilterDiscardsNegativeGain){
    auto S = single_vertex_set({6, 7}, {-2.0, -1.0});
    dps->filterCandidateSet(S);
    EXPECT_TRUE(S.segments.empty());
    EXPECT_TRUE(S.gains.empty());
}

// Selected candidates must be at least 2*max_range apart in the candidate order.
TEST_F(dp_sweeper_fixture, FilterRespectsSpacingOnIncreasingValues){
    auto S = single_vertex_set({6, 7, 8, 9, 10}, {0.0, 0.0, 2.0, 3.0, 2.0});
    dps->filterCandidateSet(S);
    ASSERT_EQ(S.segments.size(), 2u);
    EXPECT_EQ(S.segments[0].source, 8);
    EXPECT_EQ(S.segments[1].source, 10);
}

TEST_F(dp_sweeper_fixture, FilterKeepsSegmentsAndGainsAligned){
    auto S = single_vertex_set({6, 7, 8, 9, 10, 11}, {1.0, 4.0, 1.0, 1.0, 5.0, 1.0});
    dps->filterCandidateSet(S);
    ASSERT_EQ(S.segments.size(), S.gains.size());
    const std::vector<double> original = {1.0, 4.0, 1.0, 1.0, 5.0, 1.0};
    for (size_t k = 0; k < S.segments.size(); ++k)
        EXPECT_DOUBLE_EQ(S.gains[k], original[S.segments[k].source - 6]);
}

// ---------------------------------------------------------------------------
// Lexicographic choice between orientations
// ---------------------------------------------------------------------------

TEST_F(dp_sweeper_fixture, ValidChoices){
    EXPECT_TRUE(dpe->isValidChoice(0));
    EXPECT_TRUE(dpe->isValidChoice(1));
    EXPECT_FALSE(dpe->isValidChoice(-1));
    EXPECT_FALSE(dpe->isValidChoice(2));
}

TEST_F(dp_sweeper_fixture, LexicographicValuesOfSet){
    sweeper::SegmentSet S{{{6, 11}, {18, path_engine::NULL_NODE}}, {4.0, 3.0}};
    auto values = dpe->getLexicographicValues(S);
    EXPECT_DOUBLE_EQ(values.gain, 7.0);
    EXPECT_EQ(values.negative_size, -2);
    EXPECT_EQ(values.negative_length, -(dpe->geometry.cost({6, 11}).length + 1
                                       + dpe->geometry.cost({18, path_engine::NULL_NODE}).length + 1));
}

TEST_F(dp_sweeper_fixture, LexicographicValuesOfEmptySet){
    sweeper::SegmentSet S;
    auto values = dpe->getLexicographicValues(S);
    EXPECT_DOUBLE_EQ(values.gain, 0.0);
    EXPECT_EQ(values.negative_size, 0);
    EXPECT_EQ(values.negative_length, 0);
}

TEST_F(dp_sweeper_fixture, HorizontalDominatesOnHigherGain){
    std::array<sweeper::SegmentSet, 2> S{{
        {{{6, 7}}, {10.0}},
        {{{42, 47}}, {5.0}}
    }};
    EXPECT_TRUE(dpe->horizontalDominatesVertical(S));
    std::swap(S[0].gains, S[1].gains);
    EXPECT_FALSE(dpe->horizontalDominatesVertical(S));
}

TEST_F(dp_sweeper_fixture, ShorterSetDominatesOnGainTie){
    std::array<sweeper::SegmentSet, 2> S{{
        {{{6, 11}}, {5.0}},   // length 6
        {{{42, 43}}, {5.0}}   // length 2
    }};
    EXPECT_FALSE(dpe->horizontalDominatesVertical(S));
}

TEST_F(dp_sweeper_fixture, SmallerSetDominatesOnGainAndLengthTie){
    std::array<sweeper::SegmentSet, 2> S{{
        {{{6, path_engine::NULL_NODE}, {18, path_engine::NULL_NODE}}, {2.0, 3.0}},
        {{{42, 43}}, {5.0}}
    }};
    EXPECT_FALSE(dpe->horizontalDominatesVertical(S));
}

TEST_F(dp_sweeper_fixture, HorizontalDominatesOnCompleteTie){
    std::array<sweeper::SegmentSet, 2> S{{
        {{{6, 7}}, {5.0}},
        {{{42, 43}}, {5.0}}
    }};
    EXPECT_TRUE(dpe->horizontalDominatesVertical(S));
}

// ---------------------------------------------------------------------------
// selectCandidateSet / addSelectedSegmentsToSolution
// ---------------------------------------------------------------------------

TEST_F(dp_sweeper_fixture, SelectHorizontalWhenHigherGain){
    std::array<sweeper::SegmentSet, 2> S{{
        {{{6, 11}}, {10.0}},   // horizontal, higher gain
        {{{42, 47}}, {5.0}}    // vertical
    }};
    dpe->selectCandidateSet(S);
    ASSERT_FALSE(empty.path.empty());
    EXPECT_EQ(empty.path.back().source, 6);
}

TEST_F(dp_sweeper_fixture, SelectVerticalWhenHigherGain){
    std::array<sweeper::SegmentSet, 2> S{{
        {{{6, 11}}, {2.0}},
        {{{42, 47}}, {8.0}}
    }};
    dpe->selectCandidateSet(S);
    ASSERT_FALSE(empty.path.empty());
    EXPECT_EQ(empty.path.back().source, 42);
}

TEST_F(dp_sweeper_fixture, SelectChoiceCallbackOverridesGain){
    dpe->set_choice([](const std::array<sweeper::SegmentSet,2>&){ return 1; });
    std::array<sweeper::SegmentSet, 2> S{{
        {{{6, 11}}, {10.0}},   // horizontal would normally win
        {{{42, 47}}, {2.0}}
    }};
    dpe->selectCandidateSet(S);
    ASSERT_FALSE(empty.path.empty());
    EXPECT_EQ(empty.path.back().source, 42);  // vertical was forced
}

TEST_F(dp_sweeper_fixture, SelectInvalidChoiceFallsBackToDominance){
    dpe->set_choice([](const std::array<sweeper::SegmentSet,2>&){ return 7; });
    std::array<sweeper::SegmentSet, 2> S{{
        {{{6, 11}}, {2.0}},
        {{{42, 47}}, {8.0}}
    }};
    dpe->selectCandidateSet(S);
    ASSERT_FALSE(empty.path.empty());
    EXPECT_EQ(empty.path.back().source, 42);
}

TEST_F(dp_sweeper_fixture, SelectEmptySetReducesPenalty){
    double before = dpe->get_penalty();
    dpe->set_choice([](const std::array<sweeper::SegmentSet,2>&){ return 1; });
    std::array<sweeper::SegmentSet, 2> S{{
        {{{6, 11}}, {5.0}},
        {{}, {}}
    }};
    dpe->selectCandidateSet(S);
    EXPECT_TRUE(empty.path.empty());
    EXPECT_DOUBLE_EQ(dpe->get_penalty(), before / 10.0);
}

TEST_F(dp_sweeper_fixture, SelectUpdatesCoverage){
    Eigen::VectorXd before = empty.coverage;
    std::array<sweeper::SegmentSet, 2> S{{
        {{{6, 11}}, {5.0}},
        {{}, {}}
    }};
    dpe->selectCandidateSet(S);
    EXPECT_FALSE(empty.coverage.isApprox(before, 1e-9));
}

// Multiple segments in one set must all be pushed and all update coverage.
TEST_F(dp_sweeper_fixture, SelectMultipleSegmentsAllAdded){
    std::array<sweeper::SegmentSet, 2> S{{
        {{{6, 11}, {18, 23}}, {4.0, 3.0}},
        {{}, {}}
    }};
    dpe->selectCandidateSet(S);
    EXPECT_EQ(empty.path.size(), 2u);
    Eigen::VectorXd C = empty.coverage;
    dpe->coverage.reset(empty);
    EXPECT_TRUE(empty.coverage.isApprox(C, 1e-9));
}

// ---------------------------------------------------------------------------
// Sweep iterations
// ---------------------------------------------------------------------------

static void insert_initial_position(dp_sweeper_accessor& dps, cppied_solution& s, const cppied_instance& P){
    segment init = {P.initial_position, path_engine::NULL_NODE};
    dps.coverage.insert(s, init);
    s.path.push_back(init);
}

// Helper: run one sweep iteration — returns false if coverage is already satisfied.
static bool sweep_once(dp_sweeper_accessor& dps){
    dps.setUnsatisfiedCells();
    if (dps.unsatisfiedCells.empty()) return false;
    dps.addCoverageSegmentSet([](cppied_solution&){});
    return true;
}

TEST_F(dp_sweeper_fixture, UnsatisfiedCellsEmptyWhenCovered){
    dps->setUnsatisfiedCells();
    EXPECT_TRUE(dps->unsatisfiedCells.empty());
    dpe->setUnsatisfiedCells();
    EXPECT_EQ(dpe->unsatisfiedCells.size(), static_cast<size_t>(P.req.size()));
}

TEST_F(dp_sweeper_fixture, CandidateSetsAreAlignedAndOriented){
    insert_initial_position(*dpe, empty, P);
    dpe->setUnsatisfiedCells();
    auto S = dpe->constructCandidateSets();
    for (int o = 0; o < 2; ++o) {
        EXPECT_FALSE(S[o].segments.empty());
        EXPECT_EQ(S[o].segments.size(), S[o].gains.size());
        for (const auto& s : S[o].segments)
            EXPECT_EQ(dpe->geometry.is_horizontal(s), o == 0) << s;
    }
}

TEST_F(dp_sweeper_fixture, FilteredCandidateSetsArePositive){
    insert_initial_position(*dpe, empty, P);
    dpe->setUnsatisfiedCells();
    auto S = dpe->constructCandidateSets();
    dpe->filterCandidateSetsWithSpacingConstraint(S);
    for (int o = 0; o < 2; ++o) {
        EXPECT_EQ(S[o].segments.size(), S[o].gains.size());
        for (double g : S[o].gains)
            EXPECT_GT(g, 0.0);
    }
}

TEST_F(dp_sweeper_fixture, AddCoverageSegmentSetCallsHistoryOnce){
    insert_initial_position(*dpe, empty, P);
    dpe->setUnsatisfiedCells();
    int calls = 0;
    dpe->addCoverageSegmentSet([&](cppied_solution& s){ ++calls; EXPECT_EQ(&s, &empty); });
    EXPECT_EQ(calls, 1);
    EXPECT_GT(empty.path.size(), 1u);
}

// After sweeping until coverage is satisfied, the constraint must hold.
TEST_F(dp_sweeper_fixture, SweepSatisfiesCoverageConstraint){
    insert_initial_position(*dpe, empty, P);
    int max_iter = P.req.size();  // safety bound
    while (sweep_once(*dpe) && --max_iter > 0) {}

    EXPECT_TRUE((empty.coverage.array() >= P.req.array()).all());
}

// Tracked coverage must match a full reset at the end of the sweep.
TEST_F(dp_sweeper_fixture, SweepCoverageTrackingIsConsistent){
    insert_initial_position(*dpe, empty, P);
    int max_iter = P.req.size();
    while (sweep_once(*dpe) && --max_iter > 0) {}

    Eigen::VectorXd C = empty.coverage;
    dpe->coverage.reset(empty);
    EXPECT_TRUE(empty.coverage.isApprox(C, 1e-9));
}

// Each sweep iteration must strictly reduce the number of unsatisfied cells.
TEST_F(dp_sweeper_fixture, SweepMakesProgressEachIteration){
    insert_initial_position(*dpe, empty, P);
    int prev = static_cast<int>((empty.coverage.array() < P.req.array()).count());
    int iterations = 0;
    while (sweep_once(*dpe)){
        int curr = static_cast<int>((empty.coverage.array() < P.req.array()).count());
        EXPECT_LT(curr, prev) << "No progress on iteration " << iterations;
        prev = curr;
        ++iterations;
        if (iterations > static_cast<int>(P.req.size())) break;  // safety
    }
    EXPECT_GT(iterations, 0);
}

// No segment added by the sweep may be degenerate ({v, v}).
TEST_F(dp_sweeper_fixture, SweepAddsNoDegenerateSegment){
    insert_initial_position(*dpe, empty, P);
    int max_iter = P.req.size();
    while (sweep_once(*dpe) && --max_iter > 0) {}
    for (const auto& s : empty.path)
        EXPECT_NE(s.source, s.target) << s;
}

// ---------------------------------------------------------------------------
// construct
// ---------------------------------------------------------------------------

TEST_F(dp_sweeper_fixture, ConstructAlreadySatisfied){
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    int callbacks = 0;
    dps->construct([&](cppied_solution&){ ++callbacks; });
    EXPECT_EQ(callbacks, 0);  // no iterations
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

TEST_F(dp_sweeper_fixture, ConstructFromEmptyCoverageConstraint){
    dpe->construct([](cppied_solution&){});
    EXPECT_TRUE((empty.coverage.array() >= P.req.array()).all());
}

TEST_F(dp_sweeper_fixture, ConstructStartsAtInitialPosition){
    dpe->construct([](cppied_solution&){});
    ASSERT_FALSE(empty.path.empty());
    EXPECT_EQ(empty.path.front().source, P.initial_position);
}

// After construction the tracked coverage must equal the reset coverage.
TEST_F(dp_sweeper_fixture, ConstructValidCoverageTracking){
    dpe->construct([](cppied_solution&){});
    Eigen::VectorXd C = empty.coverage;
    dpe->coverage.reset(empty);
    EXPECT_TRUE(empty.coverage.isApprox(C, 1e-9));
}

// After construction, solution.cost must equal geometry.cost(solution).
TEST_F(dp_sweeper_fixture, ConstructValidCostTracking){
    dpe->construct([](cppied_solution&){});
    EXPECT_EQ(dpe->geometry.cost(empty), empty.cost);
}

// Each history callback must observe progress in coverage or a reduction of the penalty.
TEST_F(dp_sweeper_fixture, ConstructHistoryCallbackShowsProgress){
    int callbacks = 0;
    double curr_min = 0.0;
    double curr_penalty = dpe->get_penalty();
    dpe->construct([&](cppied_solution& s){
        ++callbacks;
        double new_min = s.coverage.minCoeff();
        double new_penalty = dpe->get_penalty();
        EXPECT_TRUE(new_min > curr_min || new_penalty < curr_penalty);
        curr_min = new_min;
        curr_penalty = new_penalty;
    });
    EXPECT_GT(callbacks, 0);
}

// The choice callback is consulted once per sweep iteration (history callback).
TEST_F(dp_sweeper_fixture, ConstructConsultsChoiceOncePerIteration){
    int choices = 0, callbacks = 0;
    dpe->set_choice([&](const std::array<sweeper::SegmentSet,2>&){ ++choices; return -1; });
    dpe->construct([&](cppied_solution&){ ++callbacks; });
    EXPECT_GT(callbacks, 0);
    EXPECT_EQ(choices, callbacks);
}

// ---------------------------------------------------------------------------
// Per-cell gain summation (16x16 instance, max_range > 1: each vertex observes
// several cells, so the gain of a vertex is a sum and not a single cell gain)
// ---------------------------------------------------------------------------

class dp_sweeper_large_fixture : public cppied_context_large_fixture {
protected:
    std::unique_ptr<dp_sweeper_accessor> dpe;
    cppied_solution empty;

    void SetUp() override{
        cppied_context_large_fixture::SetUp();
        empty.cost = {0,0};
        empty.coverage = Eigen::VectorXd::Zero(P.req.size());
        dpe = std::make_unique<dp_sweeper_accessor>(ctx, P, empty);
        ctx.start_time = std::chrono::high_resolution_clock::now();
    }

    double expected_vertex_gain(int v){
        double marginal_gain = 0.0;
        for (SMdIt c(P.s_pod, v); c; ++c){
            double local_gain = std::min(P.req(c.index()) - empty.coverage(c.index()), c.value());
            marginal_gain += local_gain >= 0 ? local_gain : -dpe->get_penalty();
        }
        return marginal_gain;
    }

    int positive_cells(int v){
        int count = 0;
        for (SMdIt c(P.s_pod, v); c; ++c)
            count += std::min(P.req(c.index()) - empty.coverage(c.index()), c.value()) > 0.0;
        return count;
    }

    segment row_array(int row){
        int n_cols = dpe->geometry.n_cols;
        return {row * n_cols, row * n_cols + n_cols - 1};
    }

    segment column_array(int col){
        int n_rows = dpe->geometry.n_rows;
        int first = dpe->geometry.horizontal_bound + col * n_rows;
        return {first, first + n_rows - 1};
    }

    void expect_costs_are_cell_sums(const segment& ref){
        auto costs = dpe->constructArrayCosts(ref);
        auto [u, v] = dpe->getArrayBounds(ref);
        ASSERT_EQ(costs.size(), static_cast<size_t>(v - u + 1));
        for (int k = 0; k < static_cast<int>(costs.size()); ++k) {
            double expected = expected_vertex_gain(u + k);
            if (expected > 0.0)
                EXPECT_NEAR(costs[k], expected, 1e-9) << "array " << ref << " vertex " << u + k;
            else
                EXPECT_LT(costs[k], 0.0) << "array " << ref << " vertex " << u + k;
        }
    }
};

// Guard: the instance must actually have vertices observing several unsatisfied cells.
TEST_F(dp_sweeper_large_fixture, InstanceHasMultiCellVertices){
    ASSERT_GT(P.max_range, 1);
    int multi = 0;
    for (int v = 0; v < dpe->geometry.horizontal_bound; ++v)
        multi += positive_cells(v) > 1;
    EXPECT_GT(multi, 0);
}

TEST_F(dp_sweeper_large_fixture, ArrayCostsSumCellGainsOnRows){
    for (int row : {0, dpe->geometry.n_rows / 2, dpe->geometry.n_rows})
        expect_costs_are_cell_sums(row_array(row));
}

TEST_F(dp_sweeper_large_fixture, ArrayCostsSumCellGainsOnColumns){
    for (int col : {0, dpe->geometry.n_cols / 2, dpe->geometry.n_cols})
        expect_costs_are_cell_sums(column_array(col));
}

// A vertex observing several unsatisfied cells gains strictly more than any single one of them.
TEST_F(dp_sweeper_large_fixture, ArrayCostExceedsBestSingleCellGain){
    segment ref = row_array(dpe->geometry.n_rows / 2);
    auto costs = dpe->constructArrayCosts(ref);
    auto [u, v] = dpe->getArrayBounds(ref);
    int checked = 0;
    for (int k = 0; k < static_cast<int>(costs.size()); ++k) {
        if (positive_cells(u + k) < 2) continue;
        double best_cell = 0.0;
        for (SMdIt c(P.s_pod, u + k); c; ++c)
            best_cell = std::max(best_cell, std::min(P.req(c.index()) - empty.coverage(c.index()), c.value()));
        EXPECT_GT(costs[k], best_cell) << "vertex " << u + k;
        ++checked;
    }
    EXPECT_GT(checked, 0);
}

// Partial coverage mixes satisfied (penalised) and unsatisfied cells in one vertex.
TEST_F(dp_sweeper_large_fixture, ArrayCostsSumCellGainsUnderPartialCoverage){
    segment covered = row_array(dpe->geometry.n_rows / 2);
    for (int r = 0; r < 4; ++r)
        dpe->coverage.insert(empty, covered);
    for (int row = dpe->geometry.n_rows / 2 - P.max_range; row <= dpe->geometry.n_rows / 2 + P.max_range; ++row)
        expect_costs_are_cell_sums(row_array(row));
    expect_costs_are_cell_sums(column_array(dpe->geometry.n_cols / 2));
}

// ---------------------------------------------------------------------------
// Coverage of the remaining members and edge cases
// ---------------------------------------------------------------------------

// Choice callback that forces `forced` for the first `limit` iterations and
// throws past `guard` calls, turning a non-terminating sweep into a failure.
static std::function<int(const std::array<sweeper::SegmentSet,2>&)>
bounded_choice(int forced, int limit, int guard, int& calls){
    return [forced, limit, guard, &calls](const std::array<sweeper::SegmentSet,2>&){
        if (++calls > guard) throw std::runtime_error("sweep does not terminate");
        return calls <= limit ? forced : -1;
    };
}

TEST(LexicographicSegmentSetValue, GainDominatesLengthAndSize){
    sweeper::LexicographicSegmentSetValue a{2.0, -100, -10}, b{1.0, -1, -1};
    EXPECT_GT(a, b);
}

TEST(LexicographicSegmentSetValue, LengthDominatesSize){
    sweeper::LexicographicSegmentSetValue a{1.0, -2, -10}, b{1.0, -3, -1};
    EXPECT_GT(a, b);
}

TEST(LexicographicSegmentSetValue, SizeBreaksRemainingTie){
    sweeper::LexicographicSegmentSetValue a{1.0, -2, -1}, b{1.0, -2, -2};
    EXPECT_GT(a, b);
    EXPECT_EQ(a, (sweeper::LexicographicSegmentSetValue{1.0, -2, -1}));
}

TEST_F(dp_sweeper_fixture, DefaultPenaltyAndChoice){
    EXPECT_DOUBLE_EQ(dpe->get_penalty(), 0.01);
    std::array<sweeper::SegmentSet, 2> S{{
        {{{6, 11}}, {2.0}},
        {{{42, 47}}, {8.0}}
    }};
    dpe->selectCandidateSet(S);  // default choice -1 defers to dominance
    EXPECT_EQ(empty.path.back().source, 42);
}

TEST_F(dp_sweeper_fixture, RootSegmentAddedOnEmptySolution){
    dpe->setRootSegmentIfEmptySolution();
    ASSERT_EQ(empty.path.size(), 1u);
    EXPECT_EQ(empty.path[0], (segment{P.initial_position, path_engine::NULL_NODE}));
    Eigen::VectorXd C = empty.coverage;
    dpe->coverage.reset(empty);
    EXPECT_TRUE(empty.coverage.isApprox(C, 1e-9));
}

TEST_F(dp_sweeper_fixture, RootSegmentNotAddedOnNonEmptySolution){
    auto path = sol.path;
    Eigen::VectorXd C = sol.coverage;
    dps->setRootSegmentIfEmptySolution();
    EXPECT_EQ(sol.path, path);
    EXPECT_TRUE(sol.coverage.isApprox(C, 1e-12));
}

TEST_F(dp_sweeper_fixture, AlternateOrientationsFlipsOddIndexesOnly){
    auto path = sol.path;
    dps->setInitialSolutionAlternateOrientations();
    ASSERT_EQ(sol.path.size(), path.size());
    for (size_t i = 0; i < path.size(); ++i) {
        if (i % 2 == 0)
            EXPECT_EQ(sol.path[i], path[i]) << "index " << i;
        else
            EXPECT_EQ(sol.path[i], dps->geometry.flip_segment(path[i])) << "index " << i;
        EXPECT_EQ(dps->geometry.getNonOriented(sol.path[i]), dps->geometry.getNonOriented(path[i]))
            << "index " << i;
    }
}

TEST_F(dp_sweeper_fixture, AlternateOrientationsIsAnInvolution){
    auto path = sol.path;
    dps->setInitialSolutionAlternateOrientations();
    dps->setInitialSolutionAlternateOrientations();
    EXPECT_EQ(sol.path, path);
}

TEST_F(dp_sweeper_fixture, AlternateOrientationsKeepsCoverage){
    Eigen::VectorXd C = sol.coverage;
    dps->setInitialSolutionAlternateOrientations();
    dps->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(C, 1e-9));
}

TEST_F(dp_sweeper_fixture, AlternateOrientationsOnSingleVertexPath){
    dpe->setRootSegmentIfEmptySolution();
    auto path = empty.path;
    dpe->setInitialSolutionAlternateOrientations();
    EXPECT_EQ(empty.path, path);
}

TEST_F(dp_sweeper_fixture, TspReductionKeepsFeasibilityAndCost){
    dps->solveTspReduction();
    EXPECT_EQ(dps->geometry.cost(sol), sol.cost);
    Eigen::VectorXd C = sol.coverage;
    dps->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(C, 1e-9));
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

TEST_F(dp_sweeper_fixture, TspReductionDoesNotIncreaseCost){
    cost_t before = dps->geometry.cost(sol);
    dps->solveTspReduction();
    EXPECT_LE(sol.cost, before);
}

TEST_F(dp_sweeper_fixture, ArrayBoundsOfReversedVerticalSegment){
    auto [u, v] = dpe->getArrayBounds({47, 42});
    EXPECT_EQ(u, 42);
    EXPECT_EQ(v, 47);
}

// The returned subsegment is non-oriented whatever the orientation of the array.
TEST_F(dp_sweeper_fixture, MaximumSubarrayOfReversedArrayIsNonOriented){
    for (segment ref : {segment{5, 0}, segment{47, 42}}) {
        auto [seg, gain] = dpe->maximumSubarray(ref);
        auto [fseg, fgain] = dpe->maximumSubarray(dpe->geometry.getNonOriented(ref));
        EXPECT_EQ(seg, fseg) << "from " << ref;
        EXPECT_DOUBLE_EQ(gain, fgain) << "from " << ref;
        EXPECT_EQ(seg, dpe->geometry.getNonOriented(seg)) << "from " << ref;
    }
}

// A higher penalty makes vertices observing satisfied cells less attractive.
TEST_F(dp_sweeper_fixture, ArrayCostsDecreaseWithPenalty){
    dpe->coverage.insert(empty, {6, 11});
    auto low = dpe->constructArrayCosts({0, 5});
    dpe->mPenalty = 1e3;
    auto high = dpe->constructArrayCosts({0, 5});
    ASSERT_EQ(low.size(), high.size());
    for (size_t k = 0; k < low.size(); ++k)
        EXPECT_LE(high[k], low[k]) << "offset " << k;
}

// Satisfied vertices carry a sentinel that is never selected by the filter.
TEST_F(dp_sweeper_fixture, SatisfiedArraySentinelIsFiltered){
    auto [seg, gain] = dps->maximumSubarray({6, 11});
    EXPECT_LT(gain, -1e6);
    sweeper::SegmentSet S{{seg}, {gain}};
    dps->filterCandidateSet(S);
    EXPECT_TRUE(S.segments.empty());
}

TEST_F(dp_sweeper_fixture, FilterEmptySet){
    sweeper::SegmentSet S;
    dpe->filterCandidateSet(S);
    EXPECT_TRUE(S.segments.empty());
    EXPECT_TRUE(S.gains.empty());
}

TEST_F(dp_sweeper_fixture, FilterBothOrientationsIndependently){
    std::array<sweeper::SegmentSet, 2> S{
        single_vertex_set({6, 7, 8}, {3.0, 1.0, 4.0}),
        single_vertex_set({42, 43}, {3.0, 5.0})
    };
    dpe->filterCandidateSetsWithSpacingConstraint(S);
    ASSERT_EQ(S[0].segments.size(), 2u);
    ASSERT_EQ(S[1].segments.size(), 1u);
    EXPECT_EQ(S[1].segments[0].source, 43);
}

// Candidates are subsegments of the swept arrays, inside the grid.
TEST_F(dp_sweeper_fixture, CandidateSetsStayInsideTheGrid){
    insert_initial_position(*dpe, empty, P);
    dpe->setUnsatisfiedCells();
    auto S = dpe->constructCandidateSets();
    int n_vertices = dpe->geometry.horizontal_bound + (dpe->geometry.n_cols + 1) * dpe->geometry.n_rows;
    for (int o = 0; o < 2; ++o)
        for (const auto& s : S[o].segments) {
            EXPECT_GE(s.source, 0) << s;
            EXPECT_LT(s.source, n_vertices) << s;
            if (path_engine::is_node(s.target)) {
                EXPECT_LT(s.target, n_vertices) << s;
                EXPECT_EQ(dpe->geometry.is_horizontal({s.target, path_engine::NULL_NODE}), o == 0) << s;
            }
        }
}

// The choice callback receives candidate sets that are already filtered.
TEST_F(dp_sweeper_fixture, ChoiceCallbackSeesFilteredSets){
    int calls = 0;
    dpe->set_choice([&](const std::array<sweeper::SegmentSet,2>& S){
        ++calls;
        for (int o = 0; o < 2; ++o) {
            EXPECT_EQ(S[o].segments.size(), S[o].gains.size());
            for (double g : S[o].gains) EXPECT_GT(g, 0.0);
        }
        return -1;
    });
    dpe->construct([](cppied_solution&){});
    EXPECT_GT(calls, 0);
}

// restart_backtrack forces orientations: every forced sequence must terminate.
TEST_F(dp_sweeper_fixture, ConstructTerminatesWithForcedOrientation){
    for (int forced : {0, 1}) {
        cppied_solution s;
        s.cost = {0, 0};
        s.coverage = Eigen::VectorXd::Zero(P.req.size());
        dp_sweeper_accessor d(ctx, P, s);
        int calls = 0;
        d.set_choice(bounded_choice(forced, std::numeric_limits<int>::max(), 1000, calls));
        EXPECT_NO_THROW(d.construct([](cppied_solution&){})) << "forced " << forced;
        EXPECT_TRUE((s.coverage.array() >= P.req.array()).all()) << "forced " << forced;
    }
}

// A partial path is extended, not replaced: its coverage is kept.
TEST_F(dp_sweeper_fixture, ConstructExtendsPartialPath){
    insert_initial_position(*dpe, empty, P);
    dpe->coverage.insert(empty, {6, 11});
    empty.path.push_back({6, 11});
    dpe->construct([](cppied_solution&){});
    EXPECT_TRUE((empty.coverage.array() >= P.req.array()).all());
    EXPECT_EQ(empty.path.front().source, P.initial_position);
    Eigen::VectorXd C = empty.coverage;
    dpe->coverage.reset(empty);
    EXPECT_TRUE(empty.coverage.isApprox(C, 1e-9));
}

TEST_F(dp_sweeper_fixture, ConstructTwiceIsStable){
    dpe->construct([](cppied_solution&){});
    int callbacks = 0;
    dpe->construct([&](cppied_solution&){ ++callbacks; });
    EXPECT_EQ(callbacks, 0);
    EXPECT_TRUE((empty.coverage.array() >= P.req.array()).all());
    EXPECT_EQ(dpe->geometry.cost(empty), empty.cost);
}

// A huge penalty makes every vertex observing a satisfied cell non-positive;
// the sweep must still terminate by reducing the penalty.
TEST_F(dp_sweeper_large_fixture, ConstructTerminatesWithHugePenalty){
    dpe->mPenalty = 1e6;
    int calls = 0;
    dpe->set_choice(bounded_choice(-1, 0, 10000, calls));
    EXPECT_NO_THROW(dpe->construct([](cppied_solution&){}));
    EXPECT_TRUE((empty.coverage.array() >= P.req.array()).all());
}

TEST_F(dp_sweeper_large_fixture, ConstructFeasibleAndConsistent){
    int calls = 0;
    dpe->set_choice(bounded_choice(-1, 0, 10000, calls));
    dpe->construct([](cppied_solution&){});
    EXPECT_TRUE((empty.coverage.array() >= P.req.array()).all());
    EXPECT_EQ(dpe->geometry.cost(empty), empty.cost);
    Eigen::VectorXd C = empty.coverage;
    dpe->coverage.reset(empty);
    EXPECT_TRUE(empty.coverage.isApprox(C, 1e-9));
    for (const auto& s : empty.path)
        EXPECT_NE(s.source, s.target) << s;
}

TEST_F(dp_sweeper_large_fixture, ConstructTerminatesWithForcedOrientation){
    for (int forced : {0, 1}) {
        cppied_solution s;
        s.cost = {0, 0};
        s.coverage = Eigen::VectorXd::Zero(P.req.size());
        dp_sweeper_accessor d(ctx, P, s);
        int calls = 0;
        d.set_choice(bounded_choice(forced, std::numeric_limits<int>::max(), 10000, calls));
        EXPECT_NO_THROW(d.construct([](cppied_solution&){})) << "forced " << forced;
        EXPECT_TRUE((s.coverage.array() >= P.req.array()).all()) << "forced " << forced;
    }
}
