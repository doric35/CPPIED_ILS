#include "../test_fixture.hpp"
#include "../../include/restart/restart_backtrack.hpp"

// ============================================================================
// Accessor — exposes protected members for white-box testing.
// ============================================================================

struct restart_backtrack_accessor : public restart_backtrack {
    using restart_backtrack::restart_backtrack;

    // Engines (declared in cppied_method_base)
    using cppied_method_base::geometry;
    using cppied_method_base::coverage;

    // Internal state
    using restart_backtrack::history;
    using restart_backtrack::warm_start;

    // Protected helpers
    using restart_backtrack::sample;
};

// ============================================================================
// Fixture
//
// sol       — completed, coverage-consistent 7-segment solution.
//             Used for sample() tests where path.size() must be predictable.
//
// make_empty() — returns a fresh cppied_solution with no path and zeroed
//             coverage.  Passed to restart() so that dp_sweeper always
//             constructs from scratch (history-driven decisions are active
//             from iteration 0).
//
// ctx.start_time is anchored in SetUp so that the dp_sweeper -> LKH pipeline
// receives a valid TIME_LIMIT = max_time - elapsed.
// ============================================================================
class restart_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<restart_backtrack_accessor> rb;
    cppied_solution sol;    // reference completed solution (sample tests)

    void SetUp() override {
        cppied_context_fixture::SetUp();
        rb = std::make_unique<restart_backtrack_accessor>(ctx, P);

        // Anchor the clock BEFORE any dp_sweeper / LKH call.
        ctx.start_time = std::chrono::steady_clock::now();

        sol.path = {{6, 11},
                    {23, 18},
                    {44, 42},
                    {48, 51},
                    {30, 35},
                    {75, 73},
                    {62, 63}};
        sol.cost     = {0, 0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        rb->coverage.reset(sol);
        rb->geometry.complete(sol);
        sol.cost = rb->geometry.cost(sol);
    }

    // Returns a fresh empty solution for restart() calls.
    // dp_sweeper::construct adds the initial position and runs the sweep loop
    // from scratch when the path is empty.
    [[nodiscard]] cppied_solution make_empty() const {
        cppied_solution s;
        s.coverage = Eigen::VectorXd::Zero(P.req.size());
        s.cost     = {0, 0};
        return s;
    }

    // Drains the history queue into a vector (for non-mutating inspection,
    // copy the queue first).
    static std::vector<std::vector<bool>> drain(
            std::queue<std::vector<bool>> q) {
        std::vector<std::vector<bool>> out;
        while (!q.empty()) {
            out.push_back(q.front());
            q.pop();
        }
        return out;
    }

    path_engine&     geo() { return rb->geometry; }
    coverage_engine& cov() { return rb->coverage; }
};

// ============================================================================
// sample()
//
// sample(pSol) selects floor(sqrt(path.size())) segments at random (weighted
// reservoir trick) and stores them in pSol.path.  The original path is saved
// verbatim into warm_start before the swap so that it is available as a warm
// start for the next LKH call.
// ============================================================================

// After sample the path must contain exactly floor(sqrt(original_size)) segments.
TEST_F(restart_fixture, SampleReducesPathToFloorSqrtN) {
    int n        = static_cast<int>(sol.path.size());
    int expected = static_cast<int>(std::sqrt(static_cast<double>(n)));
    rb->sample(sol);
    EXPECT_EQ(static_cast<int>(sol.path.size()), expected);
}

// Every segment kept in pSol.path after sample must have been present in the
// original path (sample never fabricates new segments).
TEST_F(restart_fixture, SampleAllSegmentsAreFromOriginalPath) {
    cppied_solution copy = sol;
    rb->sample(sol);
    for (auto& s : sol.path) {
        bool found = std::find(copy.path.begin(), copy.path.end(), s)
                     != copy.path.end();
        EXPECT_TRUE(found) << "Segment {" << s.source << "," << s.target
                           << "} not in original path";
    }
}

// A non-empty path must remain non-empty after sample (sqrt(n) >= 1 for n >= 1).
TEST_F(restart_fixture, SamplePathNonEmptyForNonEmptyInput) {
    ASSERT_GT(sol.path.size(), 0u);
    rb->sample(sol);
    EXPECT_GT(sol.path.size(), 0u);
}

// An empty path must stay empty: sqrt(0) = 0, so the selection loop runs zero
// times and new_sol stays empty.
TEST_F(restart_fixture, SampleOnEmptyPathProducesEmptyPath) {
    cppied_solution empty = make_empty();
    rb->sample(empty);
    EXPECT_EQ(empty.path.size(), 0u);
}

// ============================================================================
// restart() — first call (history empty)
//
// When history is empty restart() calls sample() to reduce the path, then
// runs dp_sweeper::construct with a choice_func that always returns -1 (free
// choice: dp_sweeper decides for every sweep iteration).
//
// The save_history callback passed to dp_sweeper appends at each iteration:
//     choices.push_back(!is_horizontal(pSol.path.back()));
//     history.push(choices);                          // push CURRENT prefix
//
// So after N sweep iterations history contains N progressively longer vectors:
//     history = { [c0], [c0,c1], [c0,c1,c2], ..., [c0,...,cN-1] }
// ============================================================================

TEST_F(restart_fixture, RestartOnEmptyHistoryDoesNotThrow) {
    cppied_solution pSol = make_empty();
    EXPECT_NO_THROW(rb->restart(pSol));
}

TEST_F(restart_fixture, RestartOnEmptyHistoryPathNonEmpty) {
    cppied_solution pSol = make_empty();
    rb->restart(pSol);
    EXPECT_GT(pSol.path.size(), 0u);
}

TEST_F(restart_fixture, RestartOnEmptyHistoryCoverageSatisfied) {
    cppied_solution pSol = make_empty();
    rb->restart(pSol);
    cov().reset(pSol);
    EXPECT_TRUE((pSol.coverage.array() >= P.req.array()).all());
}

// dp_sweeper must have run at least one sweep iteration: history cannot be
// empty after a successful construction from an empty solution.
TEST_F(restart_fixture, RestartOnEmptyHistoryPopulatesHistory) {
    cppied_solution pSol = make_empty();
    rb->restart(pSol);
    EXPECT_FALSE(rb->history.empty());
}

// ============================================================================
// restart() — history trace structure
//
// The save_history callback grows the `choices` vector by one element per
// sweep iteration and pushes a COPY into history each time.  The result is
// a sequence of strictly-increasing-length prefix vectors:
//
//   entries[0].size() == 1
//   entries[1].size() == 2
//   entries[i].size() == i + 1   for every i
//
// Each entry is a proper prefix of all later entries:
//   entries[i][j] == entries[k][j]  for all k > i, j < entries[i].size()
// ============================================================================

TEST_F(restart_fixture, HistoryShortestEntryHasLengthOne) {
    cppied_solution pSol = make_empty();
    rb->restart(pSol);
    ASSERT_FALSE(rb->history.empty());
    EXPECT_EQ(rb->history.front().size(), 1u);
}

TEST_F(restart_fixture, HistoryEntriesHaveStrictlyIncreasingLengths) {
    cppied_solution pSol = make_empty();
    rb->restart(pSol);

    auto entries = drain(rb->history);
    ASSERT_GT(entries.size(), 0u);
    for (size_t i = 1; i < entries.size(); ++i)
        EXPECT_EQ(entries[i].size(), entries[i - 1].size() + 1)
            << "Sizes not strictly increasing at position " << i;
}

TEST_F(restart_fixture, HistoryEntriesArePrefixesOfLongerEntries) {
    cppied_solution pSol = make_empty();
    rb->restart(pSol);

    auto entries = drain(rb->history);
    ASSERT_GT(entries.size(), 1u);
    for (size_t i = 1; i < entries.size(); ++i) {
        for (size_t j = 0; j < entries[i - 1].size(); ++j)
            EXPECT_EQ(entries[i][j], entries[i - 1][j])
                << "Entry " << i << " is not a prefix extension of entry "
                << (i - 1) << " at position " << j;
    }
}

TEST_F(restart_fixture, HistoryLongestEntrySizeEqualsNumberOfEntries) {
    cppied_solution pSol = make_empty();
    rb->restart(pSol);

    auto entries = drain(rb->history);
    ASSERT_FALSE(entries.empty());
    // The arithmetic: entries[0] has length 1, entries[N-1] has length N.
    EXPECT_EQ(entries.back().size(), entries.size());
}

// ============================================================================
// restart() — subsequent calls (history non-empty)
//
// When history is non-empty restart() pops history.front() (the shortest
// prefix [c0,...,cK-1]) as choices_ref and passes a choice_func to dp_sweeper
// that:
//   - At step i < K-1 : returns int(choices_ref[i])   (replay)
//   - At step i = K-1 : returns 1 - choices_ref[i]    (FLIP last choice)
//   - At step i >= K  : returns -1                     (free)
//
// This is a BFS-style systematic exploration of the dp_sweeper decision tree:
// the second call explores the subtree obtained by flipping the very first
// branching decision; the third call flips the second, etc.
// ============================================================================

TEST_F(restart_fixture, RestartFromHistoryDoesNotThrow) {
    cppied_solution pSol1 = make_empty();
    rb->restart(pSol1);           // populates history

    cppied_solution pSol2 = make_empty();
    EXPECT_NO_THROW(rb->restart(pSol2));
}

TEST_F(restart_fixture, RestartFromHistoryPathNonEmpty) {
    cppied_solution pSol1 = make_empty();
    rb->restart(pSol1);

    cppied_solution pSol2 = make_empty();
    rb->restart(pSol2);
    EXPECT_GT(pSol2.path.size(), 0u);
}

TEST_F(restart_fixture, RestartFromHistoryCoverageSatisfied) {
    cppied_solution pSol1 = make_empty();
    rb->restart(pSol1);

    cppied_solution pSol2 = make_empty();
    rb->restart(pSol2);
    cov().reset(pSol2);
    EXPECT_TRUE((pSol2.coverage.array() >= P.req.array()).all());
}

// After the second restart the shortest entry (length 1) that was at the front
// of the history queue must have been consumed: the new front must differ from
// the entry that was there before the call.
TEST_F(restart_fixture, RestartFromHistoryFrontEntryIsConsumed) {
    cppied_solution pSol1 = make_empty();
    rb->restart(pSol1);
    ASSERT_FALSE(rb->history.empty());

    std::vector<bool> old_front = rb->history.front();  // [c0]

    cppied_solution pSol2 = make_empty();
    rb->restart(pSol2);
    ASSERT_FALSE(rb->history.empty());

    // The old length-1 entry must no longer be at the front.
    EXPECT_NE(rb->history.front(), old_front);
}

// ============================================================================
// Backtracking diversity
//
// The core invariant of the backtracking scheme: the second restart flips the
// very first branching decision (c0) that was made during the first restart.
// The save_history callback records !is_horizontal(last_added_segment) after
// each sweep iteration, so the recorded c0 == true means "vertical was chosen
// at step 0", and c0 == false means "horizontal was chosen".
//
// After the second restart the new trace entries (appended at the back of the
// queue) must start with !c0, confirming that dp_sweeper took the opposite
// orientation at step 0.
//
// Test strategy:
//   1. Run first restart.  Read c0 = history.front()[0].
//   2. Run second restart.
//   3. Drain the full queue.  Length-1 entries are branch root entries; the
//      one that belongs to the second restart's trace must equal !c0.
// ============================================================================

TEST_F(restart_fixture, SecondRestartBacktracksToFlippedFirstChoice) {
    // --- First restart ---------------------------------------------------
    cppied_solution pSol1 = make_empty();
    rb->restart(pSol1);
    ASSERT_FALSE(rb->history.empty());

    // c0 is the orientation chosen at sweep iteration 0 of the first run:
    //   false → dp_sweeper chose horizontal (0)
    //   true  → dp_sweeper chose vertical   (1)
    bool c0 = rb->history.front()[0];   // shortest prefix = [c0]
    size_t history_size_after_first = rb->history.size();

    // --- Second restart --------------------------------------------------
    cppied_solution pSol2 = make_empty();
    rb->restart(pSol2);

    // --- Inspect history -------------------------------------------------
    // The history now contains:
    //   • N-1 old entries from the first restart (the length-1 entry [c0]
    //     was consumed)
    //   • M new entries from the second restart starting with [!c0]
    //
    // Drain the queue and collect all length-1 entries.  The original [c0]
    // is gone; any surviving length-1 entry belongs to the second restart's
    // trace and must have first element == !c0.
    auto entries = drain(rb->history);

    // Verify the original front entry is no longer present at length 1.
    std::vector<bool> expected_old{c0};
    bool old_still_present =
        std::find(entries.begin(), entries.end(), expected_old) != entries.end();
    EXPECT_FALSE(old_still_present)
        << "The consumed entry [c0=" << c0 << "] must not appear in history";

    //There must be no length 1 entries after the second restart, it was cosumed and not reinserted.
    auto length_one_entry =
            std::find_if(entries.begin(), entries.end(),
                         [](const std::vector<bool>& v){return v.size() <=1;});
    EXPECT_EQ(length_one_entry, entries.end());

    // There are now 2 length 2 entries with opposite first elem.
    std::vector<std::vector<bool>> new_root;
    std::copy_if(entries.begin(), entries.end(), std::back_inserter(new_root),
                 [](const std::vector<bool>& v){return v.size() == 2;});
    EXPECT_EQ(new_root.size(), 2);
    EXPECT_NE(new_root[0][0], new_root[1][0]);
}

// ============================================================================
// Multiple consecutive restarts
//
// Each restart consumes one history entry from the front and adds a new
// sub-tree to the back.  After K restarts the front entry has length K
// (the K-th entry of the original trace, whose first K-1 choices are replayed
// verbatim and the K-th is flipped on the (K+1)-th restart).
// ============================================================================

TEST_F(restart_fixture, ConsecutiveRestartsProduceNonEmptyPaths) {
    static constexpr int K = 3;
    for (int k = 0; k < K; ++k) {
        cppied_solution pSol = make_empty();
        rb->restart(pSol);
        EXPECT_GT(pSol.path.size(), 0u) << "Restart " << k << " produced empty path";
        // Re-anchor the clock between calls so that LKH always has a positive
        // time limit regardless of how long the previous restart took.
        ctx.start_time = std::chrono::steady_clock::now();
    }
}

TEST_F(restart_fixture, ConsecutiveRestartsSatisfyCoverageConstraint) {
    static constexpr int K = 3;
    for (int k = 0; k < K; ++k) {
        cppied_solution pSol = make_empty();
        rb->restart(pSol);
        cov().reset(pSol);
        EXPECT_TRUE((pSol.coverage.array() >= P.req.array()).all())
            << "Coverage constraint violated after restart " << k;
        ctx.start_time = std::chrono::steady_clock::now();
    }
}

// After K restarts the front of the history queue must be the entry of length
// K: the first K-1 entries (lengths 1, 2, ..., K-1) have been consumed one
// by one, and the K-th entry (length K) is now at the front.
TEST_F(restart_fixture, AfterKRestartsHistoryFrontHasLengthK) {
    static constexpr int K = 3;
    // First restart: populate history; the front has length 1.
    {
        cppied_solution pSol = make_empty();
        rb->restart(pSol);
        ctx.start_time = std::chrono::steady_clock::now();
    }
    // Verify the initial front length.
    ASSERT_EQ(rb->history.front().size(), 1u);

    // Restarts 2 and 3: each consumes the front and appends new entries;
    // the old front entry advances by one position in the original trace.
    for (int k = 2; k <= K; ++k) {
        cppied_solution pSol = make_empty();
        rb->restart(pSol);
        ctx.start_time = std::chrono::steady_clock::now();

        // The front of the history is now the entry of length k from the
        // original first-restart trace (entries of length 1, …, k-1 have
        // been consumed).
        EXPECT_EQ(rb->history.front().size(), static_cast<size_t>(k))
            << "After restart " << k
            << " expected history front size = " << k;
    }
}
