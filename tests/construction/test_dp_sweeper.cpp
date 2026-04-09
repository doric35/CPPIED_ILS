#include "../test_fixture.hpp"
#include "../../include/construction/dp_sweeper.hpp"

struct dp_sweeper_accessor : public dp_sweeper {
public:
    using dp_sweeper::dp_sweeper;
    using dp_sweeper::geometry;
    using dp_sweeper::coverage;
    using dp_sweeper::problem;
    using dp_sweeper::maximum_subarray;
    using dp_sweeper::filter_segment_set;
    using dp_sweeper::select;
};

class dp_sweeper_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<dp_sweeper_accessor> dps;
    cppied_solution sol;

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
        dps = std::make_unique<dp_sweeper_accessor>(ctx, P);
        dps->coverage.reset(sol);
        sol.cost = dps->geometry.cost(sol);
        ctx.start_time = std::chrono::steady_clock::now();
    }
};

TEST_F(dp_sweeper_fixture, MaximumSubarrayPositiveGainWhenUnsatisfied){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    segment ref{0, 5};  // row 0, vertices 0–5
    auto [seg, gain] = dps->maximum_subarray(empty, ref);
    EXPECT_GT(gain, 0.0);
}

TEST_F(dp_sweeper_fixture, MaximumSubarrayResultIsSubsegment){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    segment ref{0, 5};
    auto [seg, gain] = dps->maximum_subarray(empty, ref);
    int lo = path_engine::is_node(seg.target) ? std::min(seg.source, seg.target) : seg.source;
    int hi = path_engine::is_node(seg.target) ? std::max(seg.source, seg.target) : seg.source;
    EXPECT_GE(lo, 0);
    EXPECT_LE(hi, 5);
}

TEST_F(dp_sweeper_fixture, MaximumSubarraySingleVertex){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    segment ref{3, path_engine::NULL_NODE};
    auto [seg, gain] = dps->maximum_subarray(empty, ref);
    EXPECT_EQ(seg.source, 3);
    EXPECT_FALSE(path_engine::is_node(seg.target));
}

TEST_F(dp_sweeper_fixture, MaximumSubarrayNonPositiveWhenSatisfied){
    // sol has full coverage reset; row 1 (vertices 6–11) is fully covered.
    segment ref{6, 11};
    auto [seg, gain] = dps->maximum_subarray(sol, ref);
    EXPECT_LE(gain, 0.0);
}

TEST_F(dp_sweeper_fixture, MaximumSubarrayVerticalPositiveGain){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    segment ref{42, 47};  // first vertical column, vertices 42–47
    auto [seg, gain] = dps->maximum_subarray(empty, ref);
    EXPECT_GT(gain, 0.0);
    int lo = path_engine::is_node(seg.target) ? std::min(seg.source, seg.target) : seg.source;
    int hi = path_engine::is_node(seg.target) ? std::max(seg.source, seg.target) : seg.source;
    EXPECT_GE(lo, 42);
    EXPECT_LE(hi, 47);
}

TEST_F(dp_sweeper_fixture, FilterSingleSegment){
    sweeper::segment_set S{
            { {{6, 11}, 5.0} },
            0.0, true
    };
    dps->filter_segment_set(sol, S);
    ASSERT_EQ(S.S.size(), 1u);
    EXPECT_EQ(S.S[0].first.source, 6);
    EXPECT_DOUBLE_EQ(S.gain, 5.0);
}

TEST_F(dp_sweeper_fixture, FilterSpacingConstraint){
    sweeper::segment_set S{
            { {{6, path_engine::NULL_NODE}, 3.0}, {{7, path_engine::NULL_NODE}, 5.0} },
            0.0, true
    };
    dps->filter_segment_set(sol, S);
    ASSERT_EQ(S.S.size(), 1u);
    EXPECT_EQ(S.S[0].first.source, 7);  // higher gain wins
}

TEST_F(dp_sweeper_fixture, FilterRetainsCompatibleSegments){
    sweeper::segment_set S{
            { {{6, path_engine::NULL_NODE}, 3.0}, {{7, path_engine::NULL_NODE}, 1.0}, {{8, path_engine::NULL_NODE}, 4.0} },
            0.0, true
    };
    dps->filter_segment_set(sol, S);
    ASSERT_EQ(S.S.size(), 2u);
}

TEST_F(dp_sweeper_fixture, FilterDiscardsNegativeGain){
    sweeper::segment_set S{
            { {{6, path_engine::NULL_NODE}, -2.0} },
            0.0, true
    };
    dps->filter_segment_set(sol, S);
    EXPECT_TRUE(S.S.empty());
}

TEST_F(dp_sweeper_fixture, SelectHorizontalWhenHigherGain){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    std::array<sweeper::segment_set, 2> S{{
                                                  { {{{6, 11}, 10.0}}, 10.0, true  },  // horizontal, higher gain
                                                  { {{{42, 47}, 5.0}}, 5.0,  false }   // vertical
                                          }};
    dps->select(empty, S);
    ASSERT_FALSE(empty.path.empty());
    EXPECT_EQ(empty.path.back().source, 6);
}

TEST_F(dp_sweeper_fixture, SelectVerticalWhenHigherGain){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    std::array<sweeper::segment_set, 2> S{{
                                                  { {{{6, 11}, 2.0}}, 2.0, true  },
                                                  { {{{42, 47}, 8.0}}, 8.0, false }
                                          }};
    dps->select(empty, S);
    ASSERT_FALSE(empty.path.empty());
    EXPECT_EQ(empty.path.back().source, 42);
}

TEST_F(dp_sweeper_fixture, SelectChoiceCallbackOverridesGain){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    dps->set_choice([](const std::array<sweeper::segment_set,2>&){ return 1; });
    std::array<sweeper::segment_set, 2> S{{
                                                  { {{{6, 11}, 10.0}}, 10.0, true  },  // horizontal would normally win
                                                  { {{{42, 47}, 2.0}}, 2.0,  false }
                                          }};
    dps->select(empty, S);
    ASSERT_FALSE(empty.path.empty());
    EXPECT_EQ(empty.path.back().source, 42);  // vertical was forced
}

TEST_F(dp_sweeper_fixture, SelectUpdatesCoverage){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    Eigen::VectorXd before = empty.coverage;
    std::array<sweeper::segment_set, 2> S{{
                                                  { {{{6, 11}, 5.0}}, 5.0, true },
                                                  { {},                0.0, false }
                                          }};
    dps->select(empty, S);
    EXPECT_FALSE(empty.coverage.isApprox(before, 1e-9));
}

// Multiple segments in one set must all be pushed and all update coverage.
TEST_F(dp_sweeper_fixture, SelectMultipleSegmentsAllAdded){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    std::array<sweeper::segment_set, 2> S{{
                                                  { {{{6, 11}, 4.0}, {{18, 23}, 3.0}}, 7.0, true },
                                                  { {},                                 0.0, false }
                                          }};
    dps->select(empty, S);
    EXPECT_EQ(empty.path.size(), 2u);
    Eigen::VectorXd C = empty.coverage;
    dps->coverage.reset(empty);
    EXPECT_TRUE(empty.coverage.isApprox(C, 1e-9));
}

// Helper: run one sweep iteration — returns false if coverage is already satisfied.
static bool sweep_once(dp_sweeper_accessor& dps,
                       cppied_solution& sol,
                       const cppied_instance& P){
    std::vector<int> unsat;
    dps.coverage.unsatisfied_cells(sol, unsat);
    if (unsat.empty()) return false;

    iRectangle box = {std::numeric_limits<int>::max(),
                      std::numeric_limits<int>::max(),
                      std::numeric_limits<int>::min(),
                      std::numeric_limits<int>::min()};
    dps.coverage.unsatisfied_rectangle(unsat, box);

    std::array<sweeper::segment_set, 2> S{{
                                                  {{}, 0.0, true},
                                                  {{}, 0.0, false}
                                          }};
    // Horizontal: one full-row segment per row in the bounding box.
    for (int row = box.ul.y; row < box.lr.y; ++row){
        segment ref{row * dps.geometry.n_cols + box.ul.x,
                    row * dps.geometry.n_cols + box.lr.x - 1};
        if (ref.source == ref.target) ref.target = path_engine::NULL_NODE;
        S[0].S.push_back(dps.maximum_subarray(sol, ref));
    }
    // Vertical: one full-column segment per column in the bounding box.
    for (int col = box.ul.x; col < box.lr.x; ++col){
        segment ref{dps.geometry.horizontal_bound + col * dps.geometry.n_rows + box.ul.y,
                    dps.geometry.horizontal_bound + col * dps.geometry.n_rows + box.lr.y - 1};
        if (ref.source == ref.target) ref.target = path_engine::NULL_NODE;
        S[1].S.push_back(dps.maximum_subarray(sol, ref));
    }
    dps.filter_segment_set(sol, S[0]);
    dps.filter_segment_set(sol, S[1]);
    dps.select(sol, S);
    return true;
}

// After sweeping until coverage is satisfied, the constraint must hold.
TEST_F(dp_sweeper_fixture, SweepSatisfiesCoverageConstraint){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    segment init = {P.initial_position, path_engine::NULL_NODE};
    dps->coverage.insert(empty, init);
    empty.path.push_back(init);

    int max_iter = P.req.size();  // safety bound
    while (sweep_once(*dps, empty, P) && --max_iter > 0) {}

    EXPECT_TRUE((empty.coverage.array() >= P.req.array()).all());
}

// Tracked coverage must match a full reset at the end of the sweep.
TEST_F(dp_sweeper_fixture, SweepCoverageTrackingIsConsistent){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    segment init = {P.initial_position, path_engine::NULL_NODE};
    dps->coverage.insert(empty, init);
    empty.path.push_back(init);

    int max_iter = P.req.size();
    while (sweep_once(*dps, empty, P) && --max_iter > 0) {}

    Eigen::VectorXd C = empty.coverage;
    dps->coverage.reset(empty);
    EXPECT_TRUE(empty.coverage.isApprox(C, 1e-9));
}

// Each sweep iteration must strictly reduce the number of unsatisfied cells.
TEST_F(dp_sweeper_fixture, SweepMakesProgressEachIteration){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    segment init = {P.initial_position, path_engine::NULL_NODE};
    dps->coverage.insert(empty, init);
    empty.path.push_back(init);

    int prev = static_cast<int>((empty.coverage.array() < P.req.array()).count());
    int iterations = 0;
    while (sweep_once(*dps, empty, P)){
        int curr = static_cast<int>((empty.coverage.array() < P.req.array()).count());
        EXPECT_LT(curr, prev) << "No progress on iteration " << iterations;
        prev = curr;
        ++iterations;
        if (iterations > static_cast<int>(P.req.size())) break;  // safety
    }
    EXPECT_GT(iterations, 0);
}


TEST_F(dp_sweeper_fixture, ConstructAlreadySatisfied){
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    size_t initial_size = sol.path.size();
    int callbacks = 0;
    dps->construct(sol, [&](cppied_solution&){ ++callbacks; });
    EXPECT_EQ(callbacks, 0);  // no iterations
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

TEST_F(dp_sweeper_fixture, ConstructFromEmptyCoverageConstraint){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    empty.cost = {0, 0};
    dps->construct(empty, [](cppied_solution&){});
    EXPECT_TRUE((empty.coverage.array() >= P.req.array()).all());
}

// After construction the tracked coverage must equal the reset coverage.
// EXPECTED TO FAIL until Bug 1 and Bug 2 are fixed.
TEST_F(dp_sweeper_fixture, ConstructValidCoverageTracking){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    empty.cost = {0, 0};
    dps->construct(empty, [](cppied_solution&){});
    Eigen::VectorXd C = empty.coverage;
    dps->coverage.reset(empty);
    EXPECT_TRUE(empty.coverage.isApprox(C, 1e-9));
}

// After construction, pSol.cost must equal geometry.cost(pSol).
// EXPECTED TO FAIL — exposes Bug 3: cost is never updated during construction.
TEST_F(dp_sweeper_fixture, ConstructValidCostTracking){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    empty.cost = {0, 0};
    dps->construct(empty, [](cppied_solution&){});
    EXPECT_EQ(dps->geometry.cost(empty), empty.cost);
}

// Each history callback must observe progress in coverage or a reduction of the penalty.
TEST_F(dp_sweeper_fixture, ConstructHistoryCallbackShowsProgress){
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    empty.cost = {0, 0};
    int prev_unsat = static_cast<int>(P.req.size());
    int callbacks = 0;
    double curr_min = 0.0;
    double curr_penalty = dps->get_penalty();
    dps->construct(empty, [&](cppied_solution& s){
        ++callbacks;
        double new_min = s.coverage.minCoeff();
        double new_penalty = dps->get_penalty();
        EXPECT_TRUE(new_min > curr_min || new_penalty < curr_penalty);
        curr_min = new_min;
        curr_penalty = new_penalty;
    });
    EXPECT_GT(callbacks, 0);
}





