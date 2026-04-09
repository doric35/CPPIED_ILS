#include "../test_fixture.hpp"
#include "../../include/perturbations/perturbation_ri.hpp"
#include "../../include/perturbations/perturbation_pt.hpp"
#include "../../include/perturbations/perturbation_db.hpp"

// ============================================================================
// Accessors — expose protected helpers for white-box testing.
// ============================================================================

struct perturbation_ri_accessor : public perturbation_ri {
    using perturbation_ri::perturbation_ri;
    using perturbation::geometry;
    using perturbation::coverage;
    using perturbation_ri::bias_sample;
    using perturbation_ri::best_insertions;
    using perturbation_ri::ratio;
};

struct perturbation_pt_accessor : public perturbation_pt {
    using perturbation_pt::perturbation_pt;
    using perturbation::geometry;
    using perturbation::coverage;
    using perturbation_pt::uniform_sample;
    using perturbation_pt::replace;
    using perturbation_pt::replace_segment;
    using perturbation_pt::ratio;
};

struct perturbation_db_accessor : public perturbation_db {
    using perturbation_db::perturbation_db;
    using perturbation::geometry;
    using perturbation::coverage;
    using perturbation_db::uniform_sample;
    using perturbation_db::greedy_sample;
    using perturbation_db::top_6_sample;
    using perturbation_db::bridge_sample;
    using perturbation_db::double_bridge;
    using perturbation_db::gain;
    using perturbation_db::ratio;
};

// ============================================================================
// Common base fixture
//
// Provides two reference solutions that are shared by all three perturbation
// fixtures:
//
//   raw_sol   — 7 survey segments, NOT completed.  Used to test internal
//               helpers directly (bias_sample, uniform_sample, …) where a
//               small, predictable path size is desirable.
//
//   sol       — same 7 segments after geometry.complete and coverage.reset.
//               Used for d_perturbate and perturbate tests.
// ============================================================================
class perturbation_base_fixture : public cppied_context_fixture {
protected:
    cppied_solution raw_sol;    // 7 survey segments, not completed
    cppied_solution sol;        // completed, coverage-consistent

    void SetUp() override {
        cppied_context_fixture::SetUp();

        raw_sol.path = {{6, 11},
                        {23, 18},
                        {44, 42},
                        {48, 51},
                        {30, 35},
                        {75, 73},
                        {62, 63}};
        raw_sol.cost     = {0, 0};
        raw_sol.coverage = Eigen::VectorXd::Zero(P.req.size());

        // Use a temporary accessor to access the engines without constructing
        // the full algorithm.
        perturbation_ri_accessor tmp(ctx, P);
        tmp.coverage.reset(raw_sol);

        sol = raw_sol;
        tmp.geometry.complete(sol);
        tmp.coverage.reset(sol);
        sol.cost = tmp.geometry.cost(sol);
    }
};

// ============================================================================
// perturbation_ri
//
// Random Insertion (RI) perturbation:
//   bias_sample    — selects ceil(ratio*path.size()) vertices biased towards
//                    over-covered regions (reservoir weighted sampling).
//   best_insertions — for each sampled vertex, finds the cheapest insertion
//                    position and inserts a single-vertex segment there.
//   d_perturbate   — calls bias_sample then best_insertions.
//   perturbate     — calls d_perturbate then geometry.complete.
// ============================================================================
class perturbation_ri_fixture : public perturbation_base_fixture {
protected:
    std::unique_ptr<perturbation_ri_accessor> p;

    void SetUp() override {
        perturbation_base_fixture::SetUp();
        p = std::make_unique<perturbation_ri_accessor>(ctx, P);
    }

    path_engine&     geo() { return p->geometry; }
    coverage_engine& cov() { return p->coverage; }

    int expected_k() const {
        return static_cast<int>(
            std::ceil(p->ratio * static_cast<double>(raw_sol.path.size())));
    }
};

// ---------------------------------------------------------------------------
// bias_sample
// ---------------------------------------------------------------------------

// The number of sampled vertices must equal ceil(ratio * path.size()).
TEST_F(perturbation_ri_fixture, BiasSampleSizeEqualsRatioCeil) {
    std::vector<int> sampling;
    p->bias_sample(sol, sampling);
    int k = static_cast<int>(
        std::ceil(p->ratio * static_cast<double>(sol.path.size())));
    EXPECT_EQ(static_cast<int>(sampling.size()), k);
}

// Every sampled vertex ID must be a valid grid vertex index.
TEST_F(perturbation_ri_fixture, BiasSampleVerticesInValidRange) {
    std::vector<int> sampling;
    p->bias_sample(sol, sampling);
    int n_vertices = static_cast<int>(P.vertex.size());
    for (int v : sampling) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, n_vertices);
    }
}

// On a fully-covered solution every vertex has positive over-coverage, so
// bias_sample must produce a non-empty result (k >= 1 for any non-empty path).
TEST_F(perturbation_ri_fixture, BiasSampleNonEmptyOnCoveredSolution) {
    std::vector<int> sampling;
    p->bias_sample(sol, sampling);
    EXPECT_GT(sampling.size(), 0u);
}

// ---------------------------------------------------------------------------
// best_insertions
// ---------------------------------------------------------------------------

// After best_insertions the path must be strictly larger (k new segments were
// inserted).
TEST_F(perturbation_ri_fixture, BestInsertionsIncreasesPathSize) {
    cppied_solution copy = sol;
    size_t before = copy.path.size();
    std::vector<int> sampling;
    p->bias_sample(copy, sampling);
    p->best_insertions(copy, sampling);
    EXPECT_GT(copy.path.size(), before);
}

// Each inserted segment must be a single-vertex segment (path_engine::NULL_NODE or
// REVERSED_path_engine::NULL_NODE target), since best_insertions samples raw vertices.
TEST_F(perturbation_ri_fixture, BestInsertionsProducesSingleVertexSegments) {
    cppied_solution copy = sol;
    size_t before = copy.path.size();
    std::vector<int> sampling;
    p->bias_sample(copy, sampling);
    p->best_insertions(copy, sampling);
    // Count segments whose target is not a real node (inserted singles).
    int inserted_singles = 0;
    for (auto& s : copy.path)
        if (!path_engine::is_node(s.target)) ++inserted_singles;
    // There must be at least as many single-vertex segments as were inserted.
    int k = static_cast<int>(copy.path.size()) - static_cast<int>(before);
    EXPECT_GE(inserted_singles, k);
}

// The path must remain non-empty after best_insertions.
TEST_F(perturbation_ri_fixture, BestInsertionsPathNonEmpty) {
    cppied_solution copy = sol;
    std::vector<int> sampling;
    p->bias_sample(copy, sampling);
    EXPECT_NO_THROW(p->best_insertions(copy, sampling));
    EXPECT_GT(copy.path.size(), 0u);
}

// ---------------------------------------------------------------------------
// d_perturbate
// ---------------------------------------------------------------------------

TEST_F(perturbation_ri_fixture, DPerturbateDoesNotThrow) {
    EXPECT_NO_THROW(p->d_perturbate(sol));
}

// d_perturbate = bias_sample + best_insertions → path must grow.
TEST_F(perturbation_ri_fixture, DPerturbateIncreasesPathSize) {
    size_t before = sol.path.size();
    p->d_perturbate(sol);
    EXPECT_GT(sol.path.size(), before);
}

TEST_F(perturbation_ri_fixture, DPerturbatePathNonEmpty) {
    p->d_perturbate(sol);
    EXPECT_GT(sol.path.size(), 0u);
}

// ---------------------------------------------------------------------------
// perturbate (public interface: d_perturbate + geometry.complete)
// ---------------------------------------------------------------------------

TEST_F(perturbation_ri_fixture, PerturbateDoesNotThrow) {
    EXPECT_NO_THROW(p->perturbate(sol));
}

TEST_F(perturbation_ri_fixture, PerturbatePathNonEmpty) {
    p->perturbate(sol);
    EXPECT_GT(sol.path.size(), 0u);
}

// After perturbate the path is completed: cost must equal geometry.cost.
TEST_F(perturbation_ri_fixture, PerturbateCostConsistentAfterComplete) {
    p->perturbate(sol);
    sol.cost = geo().cost(sol);
    EXPECT_EQ(geo().cost(sol), sol.cost);
}

// ============================================================================
// perturbation_pt
//
// Path-perturbation (PT) perturbation:
//   uniform_sample  — selects ceil(ratio*path.size()) segment indices uniformly
//                     at random.
//   replace_segment — removes a segment from the path and replaces it with
//                     perpendicular segments covering the now-unsatisfied cells.
//   replace         — applies replace_segment to all sampled indices.
//   d_perturbate    — calls uniform_sample + replace + neighborhood_n1.
//   perturbate      — calls d_perturbate + geometry.complete.
// ============================================================================
class perturbation_pt_fixture : public perturbation_base_fixture {
protected:
    std::unique_ptr<perturbation_pt_accessor> p;

    void SetUp() override {
        perturbation_base_fixture::SetUp();
        p = std::make_unique<perturbation_pt_accessor>(ctx, P);
    }

    path_engine&     geo() { return p->geometry; }
    coverage_engine& cov() { return p->coverage; }
};

// ---------------------------------------------------------------------------
// uniform_sample
// ---------------------------------------------------------------------------

// The number of sampled indices must equal ceil(ratio * path.size()).
TEST_F(perturbation_pt_fixture, UniformSampleSizeEqualsRatioCeil) {
    std::vector<int> sampling;
    p->uniform_sample(sol, sampling);
    int k = static_cast<int>(
        std::ceil(p->ratio * static_cast<double>(sol.path.size())));
    EXPECT_EQ(static_cast<int>(sampling.size()), k);
}

// Every sampled index must be a valid position in the path.
TEST_F(perturbation_pt_fixture, UniformSampleAllIndicesInRange) {
    std::vector<int> sampling;
    p->uniform_sample(sol, sampling);
    for (int idx : sampling) {
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, static_cast<int>(sol.path.size()));
    }
}

// Sampled indices must be unique (nth_element on unique iota → no repeats).
TEST_F(perturbation_pt_fixture, UniformSampleNoDuplicates) {
    std::vector<int> sampling;
    p->uniform_sample(sol, sampling);
    std::vector<int> sorted = sampling;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end());
}

// ---------------------------------------------------------------------------
// replace_segment
// ---------------------------------------------------------------------------

// After removing segment i the tracked coverage must change (the segment was
// contributing positive coverage to at least one cell).
TEST_F(perturbation_pt_fixture, ReplaceSegmentAltersTrackedCoverage) {
    // Use a horizontal segment (index 0: {6,11}) so perpendicular replacement
    // generates vertical candidates.
    Eigen::VectorXd before_cov = sol.coverage;
    std::vector<segment> new_sol;
    p->replace_segment(sol, new_sol, 0);
    // Coverage must have changed: the segment was removed.
    EXPECT_FALSE(sol.coverage.isApprox(before_cov, 1e-9));
}

// ---------------------------------------------------------------------------
// replace
// ---------------------------------------------------------------------------

// After replace the path must be non-empty.
TEST_F(perturbation_pt_fixture, ReplacePathNonEmpty) {
    std::vector<int> sampling;
    p->uniform_sample(sol, sampling);
    p->replace(sol, sampling);
    EXPECT_GT(sol.path.size(), 0u);
}

// The sampled segments are swapped for perpendicular ones; the number of
// segments may differ from before, but the path must not disappear.
TEST_F(perturbation_pt_fixture, ReplaceDoesNotThrow) {
    std::vector<int> sampling;
    p->uniform_sample(sol, sampling);
    EXPECT_NO_THROW(p->replace(sol, sampling));
}

// ---------------------------------------------------------------------------
// d_perturbate
// ---------------------------------------------------------------------------

TEST_F(perturbation_pt_fixture, DPerturbateDoesNotThrow) {
    EXPECT_NO_THROW(p->d_perturbate(sol));
}

TEST_F(perturbation_pt_fixture, DPerturbatePathNonEmpty) {
    p->d_perturbate(sol);
    EXPECT_GT(sol.path.size(), 0u);
}

// d_perturbate calls neighborhood_n1 at the end — the solution must be
// geometrically consistent (cost computable without throwing).
TEST_F(perturbation_pt_fixture, DPerturbateCostComputableAfterCall) {
    p->d_perturbate(sol);
    EXPECT_NO_THROW({ sol.cost = geo().cost(sol); });
}

// ---------------------------------------------------------------------------
// perturbate
// ---------------------------------------------------------------------------

TEST_F(perturbation_pt_fixture, PerturbateDoesNotThrow) {
    EXPECT_NO_THROW(p->perturbate(sol));
}

TEST_F(perturbation_pt_fixture, PerturbatePathNonEmpty) {
    p->perturbate(sol);
    EXPECT_GT(sol.path.size(), 0u);
}

TEST_F(perturbation_pt_fixture, PerturbateCostConsistentAfterComplete) {
    p->perturbate(sol);
    sol.cost = geo().cost(sol);
    EXPECT_EQ(geo().cost(sol), sol.cost);
}

// ============================================================================
// perturbation_db
//
// Double-Bridge (DB) perturbation:
//   uniform_sample  — picks k path indices with a fixed parity (all even or
//                     all odd), excluding endpoints.
//   gain            — computes the transit cost saved by bridging at index i.
//   greedy_sample   — returns the index (into candidates) with the maximum
//                     gain.
//   top_6_sample    — keeps the 6 candidates closest (by geometry.dist) to the
//                     reference candidate; always includes the reference.
//   bridge_sample   — further reduces to 4 candidates via weighted random
//                     selection; always includes the reference (weight = -1).
//   double_bridge   — applies a 4-opt double-bridge rearrangement using the
//                     4 cut-point indices.
//   d_perturbate    — full pipeline; returns early (no-op) when the path is
//                     too short (k < 4 after capping by n_candidates).
//   perturbate      — calls d_perturbate + geometry.complete.
//
// Minimum path size for d_perturbate to proceed:
//   n_candidates = path.size()/2 - 2 >= 4  →  path.size() >= 12
//   k             = min(8, n_candidates)   >= 6  for top_6_sample  → path.size() >= 16
//   safe margin: path.size() >= 20 guarantees n_candidates = 8, k = 8.
// ============================================================================
class perturbation_db_fixture : public perturbation_base_fixture {
protected:
    std::unique_ptr<perturbation_db_accessor> p;
    cppied_solution large_sol;  // >= 20 single-vertex segments

    void SetUp() override {
        perturbation_base_fixture::SetUp();
        p = std::make_unique<perturbation_db_accessor>(ctx, P);

        // Build a large path of single-vertex segments so that
        // d_perturbate does not hit the early-return guard (k >= 4) and
        // top_6_sample can safely access 6 candidates (k = 8 >= 6).
        large_sol.path.clear();
        for (int i = 0; i < 20; ++i)
            large_sol.path.push_back({i, path_engine::NULL_NODE});
        large_sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        large_sol.cost     = {0, 0};
    }

    path_engine&     geo() { return p->geometry; }
    coverage_engine& cov() { return p->coverage; }

    // Convenience: collect source vertices from a path (ignoring transit).
    static std::set<int> source_set(const cppied_solution& s) {
        std::set<int> out;
        for (auto& seg : s.path)
            out.insert(seg.source);
        return out;
    }
};

// ---------------------------------------------------------------------------
// uniform_sample
// ---------------------------------------------------------------------------

// The sample must contain exactly k elements.
TEST_F(perturbation_db_fixture, UniformSampleSizeEqualsK) {
    int n_candidates = static_cast<int>(large_sol.path.size()) / 2 - 2;
    int k = static_cast<int>(
        std::max(std::log2(static_cast<double>(large_sol.path.size())), 8.0));
    k = std::min(k, n_candidates);

    std::vector<int> sampling;
    p->uniform_sample(large_sol, sampling, k);
    EXPECT_EQ(static_cast<int>(sampling.size()), k);
}

// All sampled indices must be interior positions: strictly between 0 and
// path.size()-1 (uniform_sample loops from 1 to path.size()-2).
TEST_F(perturbation_db_fixture, UniformSampleAllIndicesAreInterior) {
    std::vector<int> sampling;
    p->uniform_sample(large_sol, sampling, 8);
    for (int idx : sampling) {
        EXPECT_GT(idx, 0);
        EXPECT_LT(idx, static_cast<int>(large_sol.path.size()) - 1);
    }
}

// uniform_sample constrains all indices to have the same parity (even or odd).
TEST_F(perturbation_db_fixture, UniformSampleAllSameParity) {
    std::vector<int> sampling;
    p->uniform_sample(large_sol, sampling, 8);
    ASSERT_FALSE(sampling.empty());
    int parity = sampling[0] % 2;
    for (int idx : sampling)
        EXPECT_EQ(idx % 2, parity);
}

// No index must appear twice.
TEST_F(perturbation_db_fixture, UniformSampleNoDuplicates) {
    std::vector<int> sampling;
    p->uniform_sample(large_sol, sampling, 8);
    std::set<int> unique(sampling.begin(), sampling.end());
    EXPECT_EQ(unique.size(), sampling.size());
}

// ---------------------------------------------------------------------------
// gain
// ---------------------------------------------------------------------------

// gain computes the best transit cost joining path[i] and some other candidate.
// For a path of adjacent single-vertex segments this must be a finite cost_t.
TEST_F(perturbation_db_fixture, GainIsFiniteForValidPath) {
    std::vector<int> sampling;
    p->uniform_sample(large_sol, sampling, 8);
    ASSERT_GE(sampling.size(), 2u);

    cost_t g = p->gain(large_sol, sampling, sampling[0]);
    // gain initialises best to max_int; a finite result means at least one
    // comparison updated it.
    EXPECT_LT(g.length, std::numeric_limits<int>::max());
}

// ---------------------------------------------------------------------------
// greedy_sample
// ---------------------------------------------------------------------------

// greedy_sample returns an index into the candidates vector, not a path index.
TEST_F(perturbation_db_fixture, GreedySampleReturnsValidCandidateIndex) {
    std::vector<int> sampling;
    p->uniform_sample(large_sol, sampling, 8);
    int ref = p->greedy_sample(large_sol, sampling);
    EXPECT_GE(ref, 0);
    EXPECT_LT(ref, static_cast<int>(sampling.size()));
}

// ---------------------------------------------------------------------------
// top_6_sample
// ---------------------------------------------------------------------------

// After top_6_sample exactly 6 candidates remain (input had k=8 >= 6).
TEST_F(perturbation_db_fixture, Top6SampleReducesToExactly6) {
    std::vector<int> sampling;
    p->uniform_sample(large_sol, sampling, 8);
    int ref = p->greedy_sample(large_sol, sampling);
    p->top_6_sample(large_sol, sampling, ref);
    EXPECT_EQ(sampling.size(), 6u);
}

// ---------------------------------------------------------------------------
// bridge_sample
// ---------------------------------------------------------------------------

// After bridge_sample exactly 4 candidates remain.
TEST_F(perturbation_db_fixture, BridgeSampleReducesToExactly4) {
    std::vector<int> sampling;
    p->uniform_sample(large_sol, sampling, 8);
    int ref = p->greedy_sample(large_sol, sampling);
    p->top_6_sample(large_sol, sampling, ref);
    p->bridge_sample(large_sol, sampling, ref);
    EXPECT_EQ(sampling.size(), 4u);
}

// bridge_sample assigns weight -1 to the reference element, making it the
// first element in the ascending sort — it is always selected.
TEST_F(perturbation_db_fixture, BridgeSampleAlwaysIncludesReference) {
    std::vector<int> sampling;
    p->uniform_sample(large_sol, sampling, 8);
    int ref_idx = p->greedy_sample(large_sol, sampling);
    int ref = sampling[ref_idx];
    // Record the actual path index that ref points to (before top_6 reorders).
    // After top_6, sampling[0] = candidates[0] = path index of the element
    // that was at sampling[ref] before.
    p->top_6_sample(large_sol, sampling, ref);
    // After top_6 the reference candidate is at index 0 (cost 0 → first in
    // ascending sort).  Record it before bridge_sample changes the vector.
    int ref_path_idx = sampling[0];
    p->bridge_sample(large_sol, sampling, ref);
    bool found = std::find(sampling.begin(), sampling.end(), ref_path_idx)
                 != sampling.end();
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// double_bridge
// ---------------------------------------------------------------------------

// double_bridge must produce a non-empty path.
TEST_F(perturbation_db_fixture, DoubleBridgePathNonEmpty) {
    std::vector<int> sampling;
    p->uniform_sample(large_sol, sampling, 8);
    int ref = p->greedy_sample(large_sol, sampling);
    p->top_6_sample(large_sol, sampling, ref);
    p->bridge_sample(large_sol, sampling, ref);
    std::sort(sampling.begin(), sampling.end());
    EXPECT_NO_THROW(p->double_bridge(large_sol, sampling));
    EXPECT_GT(large_sol.path.size(), 0u);
}

// All source vertices from the original path must still be reachable in the
// rearranged path (double_bridge only reorders; it does not drop segments).
TEST_F(perturbation_db_fixture, DoubleBridgePreservesAllSourceVertices) {
    std::set<int> before = source_set(large_sol);

    std::vector<int> sampling;
    p->uniform_sample(large_sol, sampling, 8);
    int ref = p->greedy_sample(large_sol, sampling);
    p->top_6_sample(large_sol, sampling, ref);
    p->bridge_sample(large_sol, sampling, ref);
    std::sort(sampling.begin(), sampling.end());
    p->double_bridge(large_sol, sampling);

    std::set<int> after = source_set(large_sol);
    for (int v : before)
        EXPECT_TRUE(after.count(v) > 0)
            << "Source vertex " << v << " missing after double_bridge";
}

// ---------------------------------------------------------------------------
// d_perturbate — early-return guard
// ---------------------------------------------------------------------------

// When the path is too short (path.size() < 12 → n_candidates < 4 → k < 4)
// d_perturbate must not modify the path (silent no-op).
TEST_F(perturbation_db_fixture, DPerturbateEarlyReturnLeavesSmallPathUnchanged) {
    // 3-segment path: n_candidates = 3/2 - 2 = -1 < 4 → early return.
    cppied_solution small;
    small.path = {{6, 11}, {23, 18}, {44, 42}};
    small.coverage = Eigen::VectorXd::Zero(P.req.size());
    small.cost     = {0, 0};
    size_t before = small.path.size();
    p->d_perturbate(small);
    EXPECT_EQ(small.path.size(), before);
}

// ---------------------------------------------------------------------------
// d_perturbate — active path
// ---------------------------------------------------------------------------

TEST_F(perturbation_db_fixture, DPerturbateDoesNotThrowOnLargePath) {
    EXPECT_NO_THROW(p->d_perturbate(large_sol));
}

TEST_F(perturbation_db_fixture, DPerturbatePathNonEmptyOnLargePath) {
    p->d_perturbate(large_sol);
    EXPECT_GT(large_sol.path.size(), 0u);
}

// double_bridge rearranges segments but does not remove any: all original
// source vertices must still be present after d_perturbate.
TEST_F(perturbation_db_fixture, DPerturbatePreservesAllSourceVertices) {
    std::set<int> before = source_set(large_sol);
    p->d_perturbate(large_sol);
    std::set<int> after = source_set(large_sol);
    for (int v : before)
        EXPECT_TRUE(after.count(v) > 0)
            << "Source vertex " << v << " missing after d_perturbate";
}

// ---------------------------------------------------------------------------
// perturbate (public interface: d_perturbate + geometry.complete)
// ---------------------------------------------------------------------------

TEST_F(perturbation_db_fixture, PerturbateDoesNotThrow) {
    EXPECT_NO_THROW(p->perturbate(large_sol));
}

TEST_F(perturbation_db_fixture, PerturbatePathNonEmpty) {
    p->perturbate(large_sol);
    EXPECT_GT(large_sol.path.size(), 0u);
}

// After perturbate the path has been completed: geometry.cost must not throw.
TEST_F(perturbation_db_fixture, PerturbateCostComputableAfterComplete) {
    p->perturbate(large_sol);
    EXPECT_NO_THROW({ large_sol.cost = geo().cost(large_sol); });
}

// perturbate on the regular solution (7 survey segments) exercises the
// early-return path: the path may be unchanged or completed, but must remain
// non-empty and completable.
TEST_F(perturbation_db_fixture, PerturbateDoesNotThrowOnSmallSolution) {
    EXPECT_NO_THROW(p->perturbate(sol));
}

TEST_F(perturbation_db_fixture, PerturbatePathNonEmptyOnSmallSolution) {
    p->perturbate(sol);
    EXPECT_GT(sol.path.size(), 0u);
}
