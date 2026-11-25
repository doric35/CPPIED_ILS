#include "../test_fixture.hpp"
#include "../../include/perturbations/perturbation_ri.hpp"
#include "../../include/perturbations/perturbation_pt.hpp"
#include "../../include/perturbations/perturbation_db.hpp"

// ============================================================================
// Accessors — expose protected helpers for white-box testing.
//
// Since the refactor, every perturbation is bound at construction to the
// solution it perturbs (perturbation::solution); none of the helpers take a
// cppied_solution& anymore.
// ============================================================================

struct perturbation_ri_accessor : public perturbation_ri {
    using perturbation_ri::perturbation_ri;
    using perturbation::geometry;
    using perturbation::coverage;
    using perturbation::solution;
    using perturbation_ri::RATIO;
    using perturbation_ri::sampleSize;
    using perturbation_ri::insertionSample;
    using perturbation_ri::sampleCandidateUnitarySegments;
    using perturbation_ri::getCandidateWeights;
    using perturbation_ri::sampleCandidateWeight;
    using perturbation_ri::getSortedCandidates;
    using perturbation_ri::setCandidateSampling;
    using perturbation_ri::insertCandidates;
    using perturbation_ri::insertCandidate;
};

struct perturbation_pt_accessor : public perturbation_pt {
    using perturbation_pt::perturbation_pt;
    using perturbation::geometry;
    using perturbation::coverage;
    using perturbation::solution;
    using perturbation_pt::RATIO;
    using perturbation_pt::K;
    using perturbation_pt::sampleCandidates;
    using perturbation_pt::replaceCandidates;
    using perturbation_pt::replaceSegment;
    using perturbation_pt::getReplacements;
    using perturbation_pt::getHorizontalCoverReplacements;
    using perturbation_pt::getVerticalCoverReplacements;
    using perturbation_pt::setReplacementsCoverage;
    using perturbation_pt::getReplacementDirection;
    using perturbation_pt::pushBackParallelTracks;
};

struct perturbation_db_accessor : public perturbation_db {
    using perturbation_db::perturbation_db;
    using perturbation::geometry;
    using perturbation::coverage;
    using perturbation::solution;
    using perturbation_db::getDoubleBridgeIndexes;
    using perturbation_db::uniformCandidateSample;
    using perturbation_db::greedyCandidateSample;
    using perturbation_db::topSixCandidates;
    using perturbation_db::getDistancesToReferenceCandidate;
    using perturbation_db::getOrderedIndexes;
    using perturbation_db::applyDoubleBridgeMove;
    using perturbation_db::gain;
};

// ============================================================================
// Common base fixture
//
// Provides two reference solutions that are shared by all three perturbation
// fixtures:
//
//   raw_sol   — 7 survey segments, NOT completed.
//
//   sol       — same 7 segments after geometry.complete and coverage.reset.
//               Every fixture binds its perturbation to this solution.
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
        perturbation_pt_accessor tmp(ctx, P, raw_sol);
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
//   getCandidateWeights            — one weighted-reservoir (A-Res) key
//                                    log(U)/(oc+EPS) per vertex, where oc is
//                                    the vertex over-coverage.
//   getSortedCandidates            — partially sorts vertex ids so that the
//                                    sampleSize LARGEST keys come first, i.e.
//                                    sampling is biased towards vertices with
//                                    higher over-coverage.
//   setCandidateSampling           — turns the first sampleSize vertex ids into
//                                    single-vertex segments (insertionSample).
//   sampleCandidateUnitarySegments — the three steps above.
//   insertCandidate                — inserts one segment at its cheapest
//                                    position (MinimumSegmentInsertion).
//   insertCandidates               — shuffles insertionSample and inserts all.
//   d_perturbate                   — sampling + insertCandidates.
//   perturbate                     — d_perturbate + geometry.complete +
//                                    coverage.reset.
// ============================================================================
class perturbation_ri_fixture : public perturbation_base_fixture {
protected:
    std::unique_ptr<perturbation_ri_accessor> p;

    void SetUp() override {
        perturbation_base_fixture::SetUp();
        p = std::make_unique<perturbation_ri_accessor>(ctx, P, sol);
    }

    path_engine&     geo() { return p->geometry; }
    coverage_engine& cov() { return p->coverage; }

    // Over-coverage of every vertex for the bound solution.
    std::vector<double> vertex_over_coverage() {
        std::vector<double> oc(P.vertex.size(), 0.0);
        for (int v = 0; v < static_cast<int>(oc.size()); ++v) {
            segment dummy{v, path_engine::NULL_NODE};
            oc[v] = cov().over_coverage(sol, dummy);
        }
        return oc;
    }

    // Number of times each vertex is sampled over `draws` independent calls
    // to sampleCandidateUnitarySegments.
    std::vector<int> sample_counts(int draws) {
        std::vector<int> counts(P.vertex.size(), 0);
        for (int d = 0; d < draws; ++d) {
            p->insertionSample.clear();
            p->sampleCandidateUnitarySegments();
            for (auto& s : p->insertionSample)
                ++counts[s.source];
        }
        return counts;
    }
};

// ---------------------------------------------------------------------------
// constructor
// ---------------------------------------------------------------------------

// sampleSize must be ceil(RATIO * path.size()) of the bound solution.
TEST_F(perturbation_base_fixture, PerturbationRiConstructorInitialisesSampleSize) {
    perturbation_ri_accessor q(ctx, P, sol);
    auto k = static_cast<std::size_t>(
        std::ceil(q.RATIO * static_cast<double>(sol.path.size())));
    EXPECT_EQ(q.sampleSize, k);
}

// ---------------------------------------------------------------------------
// sampleCandidateWeight / getCandidateWeights
// ---------------------------------------------------------------------------

// One weight per grid vertex.
TEST_F(perturbation_ri_fixture, CandidateWeightsOnePerVertex) {
    std::vector<double> w = p->getCandidateWeights();
    EXPECT_EQ(w.size(), P.vertex.size());
}

// log(U) <= 0 and over_coverage >= 0, so every key is finite and <= 0.
TEST_F(perturbation_ri_fixture, CandidateWeightsAreFiniteAndNonPositive) {
    for (double w : p->getCandidateWeights()) {
        EXPECT_TRUE(std::isfinite(w));
        EXPECT_LE(w, 0.0);
    }
}

TEST_F(perturbation_ri_fixture, SampleCandidateWeightIsNonPositive) {
    for (int v = 0; v < static_cast<int>(P.vertex.size()); ++v)
        EXPECT_LE(p->sampleCandidateWeight(v), 0.0);
}

// ---------------------------------------------------------------------------
// getSortedCandidates
// ---------------------------------------------------------------------------

// The result must be a permutation of [0, n).
TEST_F(perturbation_ri_fixture, SortedCandidatesIsPermutation) {
    std::vector<double> w = p->getCandidateWeights();
    std::vector<int> c = p->getSortedCandidates(w);
    ASSERT_EQ(c.size(), w.size());
    std::vector<int> sorted = c;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < static_cast<int>(sorted.size()); ++i)
        EXPECT_EQ(sorted[i], i);
}

// The first sampleSize ids must hold the sampleSize LARGEST keys.
TEST_F(perturbation_ri_fixture, SortedCandidatesPrefixHoldsLargestKeys) {
    std::vector<double> w = p->getCandidateWeights();
    std::vector<int> c = p->getSortedCandidates(w);
    ASSERT_LT(p->sampleSize, c.size());
    double prefix_min = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < p->sampleSize; ++i)
        prefix_min = std::min(prefix_min, w[c[i]]);
    for (std::size_t i = p->sampleSize; i < c.size(); ++i)
        EXPECT_GE(prefix_min, w[c[i]]);
}

// Deterministic check on a hand-made key vector: the two largest keys
// (-0.1 at id 2 and -0.5 at id 0) must be selected.
TEST_F(perturbation_ri_fixture, SortedCandidatesOnKnownWeights) {
    p->sampleSize = 2;
    std::vector<double> w = {-0.5, -3.0, -0.1, -2.0, -1.0};
    std::vector<int> c = p->getSortedCandidates(w);
    std::set<int> prefix(c.begin(), c.begin() + 2);
    EXPECT_EQ(prefix, (std::set<int>{0, 2}));
}

// ---------------------------------------------------------------------------
// sampling bias — towards vertices with higher over-coverage
// ---------------------------------------------------------------------------

// For the same uniform draw U, a larger over-coverage gives a larger key
// log(U)/oc (closer to 0), so the vertex must rank earlier. Verified on keys
// built from a fixed U and known over-coverages.
TEST_F(perturbation_ri_fixture, SortedCandidatesPreferHigherOverCoverageForSameDraw) {
    const double u = 0.3;
    std::vector<double> oc = {0.5, 4.0, 1.0, 0.0, 2.0};
    std::vector<double> keys;
    for (double o : oc)
        keys.push_back(std::log(u) / (o + EPSILON));
    p->sampleSize = 2;
    std::vector<int> c = p->getSortedCandidates(keys);
    std::set<int> prefix(c.begin(), c.begin() + 2);
    EXPECT_EQ(prefix, (std::set<int>{1, 4}));   // oc = 4.0 and 2.0
}

// A vertex with zero over-coverage has key log(U)/EPS ≈ -1e7·|log U|, so it
// must never be sampled while at least sampleSize vertices have positive
// over-coverage.
// Every vertex of the fixture `sol` is over-covered, so this test uses a
// sparse solution: the same segment laid twice, which over-covers only the
// cells around it.
TEST_F(perturbation_ri_fixture, ZeroOverCoverageVerticesNeverSampled) {
    cppied_solution sparse;
    sparse.path     = {{6, 11}, {6, 11}};
    sparse.coverage = Eigen::VectorXd::Zero(P.req.size());
    sparse.cost     = {0, 0};
    cov().reset(sparse);

    perturbation_ri_accessor q(ctx, P, sparse);
    std::vector<double> oc(P.vertex.size(), 0.0);
    for (int v = 0; v < static_cast<int>(oc.size()); ++v) {
        segment dummy{v, path_engine::NULL_NODE};
        oc[v] = q.coverage.over_coverage(sparse, dummy);
    }
    std::size_t positive = std::count_if(oc.begin(), oc.end(),
                                         [](double o) { return o > 0.0; });
    ASSERT_GE(positive, q.sampleSize)
        << "fixture must have enough over-covered vertices";
    ASSERT_LT(positive, oc.size())
        << "fixture must have some vertices with zero over-coverage";

    std::vector<int> counts(oc.size(), 0);
    for (int d = 0; d < 200; ++d) {
        q.insertionSample.clear();
        q.sampleCandidateUnitarySegments();
        for (auto& s : q.insertionSample)
            ++counts[s.source];
    }
    for (int v = 0; v < static_cast<int>(oc.size()); ++v)
        if (oc[v] <= 0.0)
            EXPECT_EQ(counts[v], 0) << "vertex " << v << " has oc = 0";
}

// Statistical check: the mean over-coverage of sampled vertices must exceed
// the mean over-coverage of the over-covered vertices. With A-Res and k = 1,
// E[oc of sampled] = Σoc² / Σoc ≥ mean(oc), strictly when ocs differ.
TEST_F(perturbation_ri_fixture, SampledVerticesHaveHigherMeanOverCoverage) {
    std::vector<double> oc = vertex_over_coverage();
    double sum = 0.0, sum_sq = 0.0;
    int positive = 0;
    for (double o : oc)
        if (o > 0.0) { sum += o; sum_sq += o * o; ++positive; }
    ASSERT_GT(positive, 1);
    double mean_positive = sum / positive;
    double expected_sampled = sum_sq / sum;
    ASSERT_GT(expected_sampled, mean_positive * 1.05)
        << "fixture over-coverages too uniform for a bias test";

    p->sampleSize = 1;
    const int draws = 2000;
    std::vector<int> counts = sample_counts(draws);
    double sampled_sum = 0.0;
    for (int v = 0; v < static_cast<int>(oc.size()); ++v)
        sampled_sum += counts[v] * oc[v];
    double mean_sampled = sampled_sum / draws;

    EXPECT_GT(mean_sampled, mean_positive);
    EXPECT_NEAR(mean_sampled, expected_sampled, 0.1 * expected_sampled);
}

// The most over-covered vertex must be sampled more often than the least
// over-covered (but still positive) vertex.
TEST_F(perturbation_ri_fixture, MostOverCoveredVertexSampledMoreOften) {
    std::vector<double> oc = vertex_over_coverage();
    int v_max = -1, v_min = -1;
    for (int v = 0; v < static_cast<int>(oc.size()); ++v) {
        if (oc[v] <= 0.0) continue;
        if (v_max < 0 || oc[v] > oc[v_max]) v_max = v;
        if (v_min < 0 || oc[v] < oc[v_min]) v_min = v;
    }
    ASSERT_GE(v_max, 0);
    ASSERT_GT(oc[v_max], 2.0 * oc[v_min])
        << "fixture over-coverages too close for a bias test";

    p->sampleSize = 1;
    std::vector<int> counts = sample_counts(2000);
    EXPECT_GT(counts[v_max], counts[v_min]);
}

// ---------------------------------------------------------------------------
// setCandidateSampling / sampleCandidateUnitarySegments
// ---------------------------------------------------------------------------

TEST_F(perturbation_ri_fixture, SetCandidateSamplingBuildsUnitarySegments) {
    std::vector<int> ids = {4, 2, 7, 0};
    p->sampleSize = 3;
    p->insertionSample.clear();
    p->setCandidateSampling(ids);
    ASSERT_EQ(p->insertionSample.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(p->insertionSample[i].source, ids[i]);
        EXPECT_EQ(p->insertionSample[i].target, path_engine::NULL_NODE);
    }
}

// The number of sampled segments must equal sampleSize.
TEST_F(perturbation_ri_fixture, SampleSizeEqualsRatioCeil) {
    p->insertionSample.clear();
    p->sampleCandidateUnitarySegments();
    EXPECT_EQ(p->insertionSample.size(), p->sampleSize);
}

// Every sampled vertex id must be a valid grid vertex index, without repeats.
TEST_F(perturbation_ri_fixture, SampledVerticesValidAndDistinct) {
    p->insertionSample.clear();
    p->sampleCandidateUnitarySegments();
    std::set<int> seen;
    for (auto& s : p->insertionSample) {
        EXPECT_GE(s.source, 0);
        EXPECT_LT(s.source, static_cast<int>(P.vertex.size()));
        EXPECT_FALSE(path_engine::is_node(s.target));
        EXPECT_TRUE(seen.insert(s.source).second);
    }
}

// ---------------------------------------------------------------------------
// insertCandidate / insertCandidates
// ---------------------------------------------------------------------------

// Inserting one candidate grows the path by exactly one segment.
TEST_F(perturbation_ri_fixture, InsertCandidateGrowsPathByOne) {
    std::size_t before = sol.path.size();
    segment s{0, path_engine::NULL_NODE};
    EXPECT_NO_THROW(p->insertCandidate(s));
    EXPECT_EQ(sol.path.size(), before + 1);
}

// The inserted segment covers the requested vertex (single-vertex segment).
TEST_F(perturbation_ri_fixture, InsertCandidateInsertsSingleVertexSegment) {
    std::size_t before_singles = std::count_if(
        sol.path.begin(), sol.path.end(),
        [](const segment& s) { return !path_engine::is_node(s.target); });
    segment s{0, path_engine::NULL_NODE};
    p->insertCandidate(s);
    std::size_t after_singles = std::count_if(
        sol.path.begin(), sol.path.end(),
        [](const segment& s) { return !path_engine::is_node(s.target); });
    EXPECT_EQ(after_singles, before_singles + 1);
}

// insertCandidate keeps the tracked coverage consistent with the path.
TEST_F(perturbation_ri_fixture, InsertCandidateKeepsCoverageConsistent) {
    segment s{0, path_engine::NULL_NODE};
    p->insertCandidate(s);
    cppied_solution check = sol;
    cov().reset(check);
    EXPECT_TRUE(sol.coverage.isApprox(check.coverage, 1e-9));
}

// insertCandidates inserts every sampled segment.
TEST_F(perturbation_ri_fixture, InsertCandidatesGrowsPathBySampleSize) {
    std::size_t before = sol.path.size();
    p->insertionSample.clear();
    p->sampleCandidateUnitarySegments();
    std::size_t k = p->insertionSample.size();
    EXPECT_NO_THROW(p->insertCandidates());
    EXPECT_EQ(sol.path.size(), before + k);
}

// ---------------------------------------------------------------------------
// d_perturbate
// ---------------------------------------------------------------------------

TEST_F(perturbation_ri_fixture, DPerturbateDoesNotThrow) {
    EXPECT_NO_THROW(p->d_perturbate());
}

// d_perturbate = sampling + insertCandidates → path must grow.
TEST_F(perturbation_ri_fixture, DPerturbateIncreasesPathSize) {
    std::size_t before = sol.path.size();
    p->d_perturbate();
    EXPECT_GT(sol.path.size(), before);
}

// insertionSample is cleared on each call: calling twice must not accumulate.
TEST_F(perturbation_ri_fixture, DPerturbateResetsInsertionSample) {
    p->d_perturbate();
    p->d_perturbate();
    EXPECT_EQ(p->insertionSample.size(), p->sampleSize);
}

// ---------------------------------------------------------------------------
// perturbate (public interface: d_perturbate + complete + reset)
// ---------------------------------------------------------------------------

TEST_F(perturbation_ri_fixture, PerturbateDoesNotThrow) {
    EXPECT_NO_THROW(p->perturbate());
}

TEST_F(perturbation_ri_fixture, PerturbatePathNonEmpty) {
    p->perturbate();
    EXPECT_GT(sol.path.size(), 0u);
}

// After perturbate the path is completed: cost must equal geometry.cost.
TEST_F(perturbation_ri_fixture, PerturbateCostConsistentAfterComplete) {
    p->perturbate();
    sol.cost = geo().cost(sol);
    EXPECT_EQ(geo().cost(sol), sol.cost);
}

// perturbate ends with coverage.reset: tracked coverage matches the path.
TEST_F(perturbation_ri_fixture, PerturbateCoverageConsistent) {
    p->perturbate();
    cppied_solution check = sol;
    cov().reset(check);
    EXPECT_TRUE(sol.coverage.isApprox(check.coverage, 1e-9));
}

// ============================================================================
// perturbation_pt
//
// Parallel-Tracks (PT) perturbation:
//   sampleCandidates               — K = ceil(RATIO*path.size()) distinct path
//                                    indices in [1, path.size()), sorted.
//   getHorizontalCoverReplacements — one horizontal track per row around box.
//   getVerticalCoverReplacements   — one vertical track per column around box.
//   getReplacements                — tracks perpendicular to s covering the
//                                    cells left unsatisfied once s is removed.
//   setReplacementsCoverage        — adds the tracks to the tracked coverage.
//   getReplacementDirection        — orientation of the first track that is
//                                    cheapest to reach from new_sol.back().
//   pushBackParallelTracks         — appends the tracks, alternating direction
//                                    (boustrophedon).
//   replaceSegment                 — remove + replacements + push.
//   replaceCandidates              — replaces every sampled index, keeping all
//                                    other segments in order.
//   d_perturbate                   — sample + replace + neighborhood_trim.
//   perturbate                     — d_perturbate + complete + reset.
// ============================================================================
class perturbation_pt_fixture : public perturbation_base_fixture {
protected:
    std::unique_ptr<perturbation_pt_accessor> p;

    void SetUp() override {
        perturbation_base_fixture::SetUp();
        p = std::make_unique<perturbation_pt_accessor>(ctx, P, sol);
    }

    path_engine&     geo() { return p->geometry; }
    coverage_engine& cov() { return p->coverage; }

    // Index (>= 1) of the first survey segment of `sol` with the given
    // orientation whose removal leaves at least one cell unsatisfied, i.e. a
    // segment that actually needs replacement tracks. -1 if none.
    int first_index(bool horizontal) {
        for (int i = 1; i < static_cast<int>(sol.path.size()); ++i) {
            const segment& s = sol.path[i];
            if (!path_engine::is_node(s.target) ||
                geo().is_horizontal(s) != horizontal)
                continue;
            cov().remove(sol, s);
            std::vector<int> unsat;
            cov().unsatisfied_cells(sol, s, unsat);
            cov().insert(sol, s);
            if (!unsat.empty())
                return i;
        }
        return -1;
    }

    // Tracked coverage equals the coverage recomputed from the path.
    ::testing::AssertionResult coverage_consistent(const cppied_solution& s) {
        cppied_solution check = s;
        cov().reset(check);
        if (s.coverage.isApprox(check.coverage, 1e-9))
            return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure()
               << "tracked coverage differs from recomputed coverage";
    }
};

// ---------------------------------------------------------------------------
// constructor
// ---------------------------------------------------------------------------

// K must be ceil(RATIO * path.size()) of the bound solution.
TEST_F(perturbation_base_fixture, PerturbationPtConstructorInitialisesK) {
    perturbation_pt_accessor q(ctx, P, sol);
    auto k = static_cast<std::size_t>(
        std::ceil(q.RATIO * static_cast<double>(sol.path.size())));
    EXPECT_EQ(q.K, k);
}

// ---------------------------------------------------------------------------
// sampleCandidates
// ---------------------------------------------------------------------------

TEST_F(perturbation_pt_fixture, SampleCandidatesSizeEqualsK) {
    EXPECT_EQ(p->sampleCandidates().size(), p->K);
}

// Indices are in [1, path.size()) — index 0 is never replaced.
TEST_F(perturbation_pt_fixture, SampleCandidatesIndicesInRange) {
    p->K = sol.path.size() - 1;
    for (int idx : p->sampleCandidates()) {
        EXPECT_GE(idx, 1);
        EXPECT_LT(idx, static_cast<int>(sol.path.size()));
    }
}

TEST_F(perturbation_pt_fixture, SampleCandidatesSortedAndDistinct) {
    p->K = sol.path.size() / 2;
    std::vector<int> c = p->sampleCandidates();
    EXPECT_TRUE(std::is_sorted(c.begin(), c.end()));
    EXPECT_EQ(std::adjacent_find(c.begin(), c.end()), c.end());
}

// ---------------------------------------------------------------------------
// getHorizontalCoverReplacements / getVerticalCoverReplacements
// ---------------------------------------------------------------------------

TEST_F(perturbation_pt_fixture, HorizontalCoverReplacementsAreHorizontalOnePerRow) {
    iRectangle box;
    box.ul = {1, 2};
    box.lr = {4, 3};
    std::vector<segment> r = p->getHorizontalCoverReplacements(box);
    int first = std::max(0, box.ul.y - (P.max_range - 1));
    int last  = std::min(geo().n_rows, box.lr.y + (P.max_range - 1));
    ASSERT_EQ(static_cast<int>(r.size()), last - first + 1);
    for (int k = 0; k < static_cast<int>(r.size()); ++k) {
        EXPECT_EQ(r[k].source, (first + k) * geo().n_cols + box.ul.x);
        EXPECT_TRUE(geo().is_horizontal(r[k]));
    }
}

TEST_F(perturbation_pt_fixture, VerticalCoverReplacementsAreVerticalOnePerColumn) {
    iRectangle box;
    box.ul = {1, 2};
    box.lr = {3, 5};
    std::vector<segment> r = p->getVerticalCoverReplacements(box);
    int first = std::max(0, box.ul.x - (P.max_range - 1));
    int last  = std::min(geo().n_cols, box.lr.x + (P.max_range - 1));
    ASSERT_EQ(static_cast<int>(r.size()), last - first + 1);
    for (int k = 0; k < static_cast<int>(r.size()); ++k) {
        EXPECT_EQ(r[k].source,
                  geo().horizontal_bound + (first + k) * geo().n_rows + box.ul.y);
        EXPECT_FALSE(geo().is_horizontal(r[k]));
    }
}

// ---------------------------------------------------------------------------
// getReplacements
// ---------------------------------------------------------------------------

// Removing a horizontal segment yields vertical replacement tracks.
TEST_F(perturbation_pt_fixture, ReplacementsOfHorizontalSegmentAreVertical) {
    int i = first_index(true);
    ASSERT_GE(i, 0);
    segment s = sol.path[i];
    cov().remove(sol, s);
    std::vector<segment> r = p->getReplacements(s);
    ASSERT_FALSE(r.empty());
    for (auto& t : r)
        EXPECT_FALSE(geo().is_horizontal(t));
}

// Removing a vertical segment yields horizontal replacement tracks.
TEST_F(perturbation_pt_fixture, ReplacementsOfVerticalSegmentAreHorizontal) {
    int i = first_index(false);
    ASSERT_GE(i, 0);
    segment s = sol.path[i];
    cov().remove(sol, s);
    std::vector<segment> r = p->getReplacements(s);
    ASSERT_FALSE(r.empty());
    for (auto& t : r)
        EXPECT_TRUE(geo().is_horizontal(t));
}

// The replacement tracks must re-satisfy every cell left unsatisfied by the
// removal.
TEST_F(perturbation_pt_fixture, ReplacementsCoverUnsatisfiedCells) {
    int i = first_index(true);
    ASSERT_GE(i, 0);
    segment s = sol.path[i];
    cov().remove(sol, s);
    std::vector<int> unsat;
    cov().unsatisfied_cells(sol, s, unsat);
    ASSERT_FALSE(unsat.empty());

    p->setReplacementsCoverage(p->getReplacements(s));
    for (int c : unsat)
        EXPECT_GE(sol.coverage(c), P.req(c)) << "cell " << c;
}

// A redundant segment (all its cells stay satisfied once removed) has no
// replacement tracks.
TEST_F(perturbation_pt_fixture, ReplacementsOfRedundantSegmentAreEmpty) {
    int i = first_index(true);
    ASSERT_GE(i, 0);
    segment s = sol.path[i];
    cov().insert(sol, s);        // s is now covered twice
    cov().remove(sol, s);        // remove one copy
    EXPECT_TRUE(p->getReplacements(s).empty());
}

// ---------------------------------------------------------------------------
// setReplacementsCoverage
// ---------------------------------------------------------------------------

TEST_F(perturbation_pt_fixture, SetReplacementsCoverageAddsEveryTrack) {
    int i = first_index(true);
    ASSERT_GE(i, 0);
    segment s = sol.path[i];
    cov().remove(sol, s);
    std::vector<segment> r = p->getReplacements(s);

    cppied_solution expected = sol;
    for (auto& t : r)
        cov().insert(expected, t);
    p->setReplacementsCoverage(r);
    EXPECT_TRUE(sol.coverage.isApprox(expected.coverage, 1e-9));
}

// ---------------------------------------------------------------------------
// getReplacementDirection
// ---------------------------------------------------------------------------

// The direction is one of the two orientations of the first track, and the
// first track oriented that way is not more expensive to reach than the other
// orientation.
TEST_F(perturbation_pt_fixture, ReplacementDirectionIsCheapestOrientationOfFirstTrack) {
    int i = first_index(true);
    ASSERT_GE(i, 1);
    segment s = sol.path[i];
    cov().remove(sol, s);
    std::vector<segment> r = p->getReplacements(s);
    ASSERT_FALSE(r.empty());
    std::vector<segment> new_sol(sol.path.begin(), sol.path.begin() + i);

    segment front = r.front();
    segment flipped = geo().flip_segment(front);
    for (int trial = 0; trial < 20; ++trial) {
        direction d = p->getReplacementDirection(new_sol, r);
        ASSERT_TRUE(d == geo().get_direction(front) ||
                    d == geo().get_direction(flipped));
        segment chosen = d == geo().get_direction(front) ? front : flipped;
        segment other  = d == geo().get_direction(front) ? flipped : front;
        EXPECT_LE(geo().dist(new_sol.back(), chosen),
                  geo().dist(new_sol.back(), other));
    }
}

// ---------------------------------------------------------------------------
// pushBackParallelTracks
// ---------------------------------------------------------------------------

TEST_F(perturbation_pt_fixture, PushBackParallelTracksAppendsEveryTrack) {
    iRectangle box;
    box.ul = {1, 2};
    box.lr = {4, 3};
    std::vector<segment> r = p->getHorizontalCoverReplacements(box);
    std::vector<segment> new_sol = {sol.path[0]};
    p->pushBackParallelTracks(new_sol, r, direction::E);
    EXPECT_EQ(new_sol.size(), 1 + r.size());
}

TEST_F(perturbation_pt_fixture, PushBackParallelTracksFirstTrackHasGivenDirection) {
    iRectangle box;
    box.ul = {1, 2};
    box.lr = {4, 3};
    std::vector<segment> r = p->getHorizontalCoverReplacements(box);
    std::vector<segment> new_sol;
    p->pushBackParallelTracks(new_sol, r, direction::W);
    ASSERT_FALSE(new_sol.empty());
    EXPECT_EQ(geo().get_direction(new_sol.front()), direction::W);
}

// Consecutive tracks must alternate direction (boustrophedon pattern).
TEST_F(perturbation_pt_fixture, PushBackParallelTracksAlternatesDirection) {
    iRectangle box;
    box.ul = {1, 2};
    box.lr = {4, 3};
    std::vector<segment> r = p->getHorizontalCoverReplacements(box);
    ASSERT_GE(r.size(), 2u);
    std::vector<segment> new_sol;
    p->pushBackParallelTracks(new_sol, r, direction::E);
    for (std::size_t k = 1; k < new_sol.size(); ++k)
        EXPECT_EQ(geo().get_direction(new_sol[k]),
                  flip_direction(geo().get_direction(new_sol[k - 1])))
            << "tracks " << k - 1 << " and " << k;
}

// ---------------------------------------------------------------------------
// replaceSegment
// ---------------------------------------------------------------------------

// After removing segment i the tracked coverage must change.
TEST_F(perturbation_pt_fixture, ReplaceSegmentAltersTrackedCoverage) {
    int i = first_index(true);
    ASSERT_GE(i, 1);
    Eigen::VectorXd before_cov = sol.coverage;
    std::vector<segment> new_sol(sol.path.begin(), sol.path.begin() + i);
    p->replaceSegment(new_sol, i);
    EXPECT_FALSE(sol.coverage.isApprox(before_cov, 1e-9));
}

// Every track pushed by replaceSegment is perpendicular to the replaced one.
TEST_F(perturbation_pt_fixture, ReplaceSegmentPushesPerpendicularTracks) {
    int i = first_index(true);
    ASSERT_GE(i, 1);
    std::vector<segment> new_sol(sol.path.begin(), sol.path.begin() + i);
    std::size_t before = new_sol.size();
    p->replaceSegment(new_sol, i);
    ASSERT_GT(new_sol.size(), before);
    for (std::size_t k = before; k < new_sol.size(); ++k)
        EXPECT_FALSE(geo().is_horizontal(new_sol[k]));
}

// Replacing a redundant segment pushes nothing and leaves coverage at the
// value it had without that segment.
TEST_F(perturbation_pt_fixture, ReplaceSegmentOfRedundantSegmentPushesNothing) {
    int i = first_index(true);
    ASSERT_GE(i, 1);
    sol.path.insert(sol.path.begin() + i + 1, sol.path[i]);   // duplicate
    cov().reset(sol);
    std::vector<segment> new_sol(sol.path.begin(), sol.path.begin() + i + 1);
    std::size_t before = new_sol.size();
    EXPECT_NO_THROW(p->replaceSegment(new_sol, i + 1));
    EXPECT_EQ(new_sol.size(), before);
}

// ---------------------------------------------------------------------------
// replaceCandidates
// ---------------------------------------------------------------------------

// Segments that were not sampled must all be kept, in order, including
// those after the last sampled index.
TEST_F(perturbation_pt_fixture, ReplaceCandidatesKeepsNonSampledSegments) {
    int i = first_index(true);
    ASSERT_GE(i, 1);
    ASSERT_LT(i, static_cast<int>(sol.path.size()) - 1)
        << "need segments after the sampled index";
    std::vector<segment> kept;
    for (int k = 0; k < static_cast<int>(sol.path.size()); ++k)
        if (k != i) kept.push_back(sol.path[k]);

    p->replaceCandidates({i});

    auto same = [](const segment& a, const segment& b) {
        return a.source == b.source && a.target == b.target;
    };
    // `kept` must be a subsequence of the new path.
    std::size_t j = 0;
    for (auto& s : sol.path)
        if (j < kept.size() && same(s, kept[j])) ++j;
    EXPECT_EQ(j, kept.size()) << "segments after the last candidate were dropped";
}

// The last segment of the path is kept when it is not sampled.
TEST_F(perturbation_pt_fixture, ReplaceCandidatesKeepsLastSegment) {
    int i = first_index(true);
    ASSERT_GE(i, 1);
    ASSERT_LT(i, static_cast<int>(sol.path.size()) - 1);
    segment last = sol.path.back();
    p->replaceCandidates({i});
    ASSERT_FALSE(sol.path.empty());
    EXPECT_EQ(sol.path.back().source, last.source);
    EXPECT_EQ(sol.path.back().target, last.target);
}

// Tracked coverage stays consistent with the path.
TEST_F(perturbation_pt_fixture, ReplaceCandidatesKeepsCoverageConsistent) {
    p->replaceCandidates(p->sampleCandidates());
    EXPECT_TRUE(coverage_consistent(sol));
}

// Replacing nothing leaves the path untouched.
TEST_F(perturbation_pt_fixture, ReplaceCandidatesWithNoCandidateIsNoOp) {
    std::size_t before = sol.path.size();
    p->replaceCandidates({});
    EXPECT_EQ(sol.path.size(), before);
}

// ---------------------------------------------------------------------------
// d_perturbate
// ---------------------------------------------------------------------------

TEST_F(perturbation_pt_fixture, DPerturbateDoesNotThrow) {
    EXPECT_NO_THROW(p->d_perturbate());
}

TEST_F(perturbation_pt_fixture, DPerturbatePathNonEmpty) {
    p->d_perturbate();
    EXPECT_GT(sol.path.size(), 0u);
}

// d_perturbate runs neighborhood_trim at the end — the solution must be
// geometrically consistent (cost computable without throwing).
TEST_F(perturbation_pt_fixture, DPerturbateCostComputableAfterCall) {
    p->d_perturbate();
    EXPECT_NO_THROW({ sol.cost = geo().cost(sol); });
}

// ---------------------------------------------------------------------------
// perturbate
// ---------------------------------------------------------------------------

TEST_F(perturbation_pt_fixture, PerturbateDoesNotThrow) {
    EXPECT_NO_THROW(p->perturbate());
}

TEST_F(perturbation_pt_fixture, PerturbatePathNonEmpty) {
    p->perturbate();
    EXPECT_GT(sol.path.size(), 0u);
}

TEST_F(perturbation_pt_fixture, PerturbateCostConsistentAfterComplete) {
    p->perturbate();
    sol.cost = geo().cost(sol);
    EXPECT_EQ(geo().cost(sol), sol.cost);
}

// perturbate keeps the solution feasible: every cell stays satisfied.
TEST_F(perturbation_pt_fixture, PerturbateKeepsCoverageSatisfied) {
    ASSERT_TRUE(cov().satisfy(sol));
    p->perturbate();
    EXPECT_TRUE(cov().satisfy(sol));
}

// ============================================================================
// perturbation_db
//
// Double-Bridge (DB) perturbation:
//   uniformCandidateSample           — k = min(max(ln(path.size()/2), 8), #cands)
//                                      shuffled interior indices of one parity.
//   gain                             — best transit cost joining path[i] to
//                                      another sampled index.
//   greedyCandidateSample            — the PATH INDEX with the maximum gain.
//   getDistancesToReferenceCandidate — geometry.dist(path[ref], path[i]) for
//                                      every candidate (0 for ref itself).
//   getOrderedIndexes                — positions sorted by distance.
//   topSixCandidates                 — the 6 candidates closest to ref.
//   getDoubleBridgeIndexes           — ref + 3 random of the other 5, sorted.
//   applyDoubleBridgeMove            — blocks A|B|C|D|E cut after the 4
//                                      indexes are reordered A D C B E (linked).
//   d_perturbate                     — getDoubleBridgeIndexes + apply.
//   perturbate                       — d_perturbate + complete + reset.
//
// Two instances are used: `p` bound to the completed 7-segment `sol`, and `pl`
// bound to `large_sol` (20 single-vertex segments → 9 candidates, k = 8).
// ============================================================================
class perturbation_db_fixture : public perturbation_base_fixture {
protected:
    std::unique_ptr<perturbation_db_accessor> p;   // bound to sol
    std::unique_ptr<perturbation_db_accessor> pl;  // bound to large_sol
    cppied_solution large_sol;  // 20 single-vertex segments

    void SetUp() override {
        perturbation_base_fixture::SetUp();
        p = std::make_unique<perturbation_db_accessor>(ctx, P, sol);
        large_sol = single_vertex_path(20);
        pl = std::make_unique<perturbation_db_accessor>(ctx, P, large_sol);
    }

    path_engine&     geo() { return p->geometry; }
    coverage_engine& cov() { return p->coverage; }

    cppied_solution single_vertex_path(int n) {
        cppied_solution s;
        for (int i = 0; i < n; ++i)
            s.path.push_back({i, path_engine::NULL_NODE});
        s.coverage = Eigen::VectorXd::Zero(P.req.size());
        s.cost     = {0, 0};
        return s;
    }

    static std::set<int> source_set(const cppied_solution& s) {
        std::set<int> out;
        for (auto& seg : s.path)
            out.insert(seg.source);
        return out;
    }

    // Vertices visited by a path (sources, targets and every node in between).
    std::set<int> node_set(cppied_solution& s) {
        std::set<int> out;
        auto add_node = [&](cppied_solution&, int v){ out.insert(v); };
        for (auto& seg : s.path)
            p->coverage.apply(s, seg, add_node);
        return out;
    }

    // Every original vertex is still visited after the move.
    ::testing::AssertionResult keeps_vertices(const std::set<int>& before,
                                              cppied_solution& s) {
        std::set<int> sources = source_set(s);
        std::set<int> nodes = node_set(s);
        for (int v : before)
            if (!sources.count(v) && !nodes.count(v))
                return ::testing::AssertionFailure() << "vertex " << v << " lost";
        return ::testing::AssertionSuccess();
    }
};

// ---------------------------------------------------------------------------
// uniformCandidateSample
// ---------------------------------------------------------------------------

// 20 segments → interior indices 1..18 → 9 of each parity → k = min(8, 9).
TEST_F(perturbation_db_fixture, UniformCandidateSampleSizeIsK) {
    int k = static_cast<int>(std::max(std::log(large_sol.path.size() / 2.0), 8.0));
    EXPECT_EQ(static_cast<int>(pl->uniformCandidateSample().size()), std::min(k, 9));
}

// The sample is capped by the number of same-parity interior indices.
TEST_F(perturbation_db_fixture, UniformCandidateSampleCappedByCandidates) {
    cppied_solution s = single_vertex_path(10);   // interior 1..8 → 4 per parity
    perturbation_db_accessor q(ctx, P, s);
    EXPECT_EQ(q.uniformCandidateSample().size(), 4u);
}

TEST_F(perturbation_db_fixture, UniformCandidateSampleAllIndicesAreInterior) {
    for (int idx : pl->uniformCandidateSample()) {
        EXPECT_GT(idx, 0);
        EXPECT_LT(idx, static_cast<int>(large_sol.path.size()) - 1);
    }
}

TEST_F(perturbation_db_fixture, UniformCandidateSampleAllSameParity) {
    std::vector<int> c = pl->uniformCandidateSample();
    ASSERT_FALSE(c.empty());
    for (int idx : c)
        EXPECT_EQ(idx % 2, c[0] % 2);
}

TEST_F(perturbation_db_fixture, UniformCandidateSampleNoDuplicates) {
    std::vector<int> c = pl->uniformCandidateSample();
    std::set<int> unique(c.begin(), c.end());
    EXPECT_EQ(unique.size(), c.size());
}

// ---------------------------------------------------------------------------
// gain / greedyCandidateSample
// ---------------------------------------------------------------------------

TEST_F(perturbation_db_fixture, GainIsFiniteForValidPath) {
    std::vector<int> c = pl->uniformCandidateSample();
    ASSERT_GE(c.size(), 2u);
    cost_t g = pl->gain(c, c[0]);
    EXPECT_LT(g.length, std::numeric_limits<int>::max());
}

// greedyCandidateSample returns a path index taken from the candidates.
TEST_F(perturbation_db_fixture, GreedyCandidateSampleReturnsACandidate) {
    std::vector<int> c = pl->uniformCandidateSample();
    int ref = pl->greedyCandidateSample(c);
    EXPECT_NE(std::find(c.begin(), c.end(), ref), c.end());
}

// ... and it is the candidate with the maximum gain.
TEST_F(perturbation_db_fixture, GreedyCandidateSampleMaximisesGain) {
    std::vector<int> c = pl->uniformCandidateSample();
    int ref = pl->greedyCandidateSample(c);
    cost_t best = pl->gain(c, ref);
    for (int i : c)
        EXPECT_FALSE(best < pl->gain(c, i)) << "candidate " << i;
}

// ---------------------------------------------------------------------------
// getDistancesToReferenceCandidate / getOrderedIndexes
// ---------------------------------------------------------------------------

TEST_F(perturbation_db_fixture, DistancesToReferenceMatchGeometry) {
    std::vector<int> c = pl->uniformCandidateSample();
    int ref = c[0];
    std::vector<cost_t> d = pl->getDistancesToReferenceCandidate(c, ref);
    ASSERT_EQ(d.size(), c.size());
    EXPECT_EQ(d[0], (cost_t{0, 0}));
    for (std::size_t k = 1; k < c.size(); ++k)
        EXPECT_EQ(d[k], geo().dist(large_sol.path[ref], large_sol.path[c[k]]));
}

// ---------------------------------------------------------------------------
// No candidate other than the reference is at distance {0,0} from it, so the
// reference is always first after sorting (std::sort is enough).
// ---------------------------------------------------------------------------

// Any two distinct same-parity interior indices can be in one candidate set:
// check every such pair, in both directions.
static ::testing::AssertionResult no_zero_distance_candidates(path_engine& g,
                                                              const cppied_solution& s) {
    const int n = static_cast<int>(s.path.size());
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            if (i != j && (i - j) % 2 == 0 &&
                g.dist(s.path[i], s.path[j]) == cost_t{0, 0})
                return ::testing::AssertionFailure()
                       << "dist(path[" << i << "], path[" << j << "]) == {0,0}";
    return ::testing::AssertionSuccess();
}

TEST_F(perturbation_db_fixture, NoZeroDistanceCandidatesOnSinglePath) {
    EXPECT_TRUE(no_zero_distance_candidates(geo(), large_sol));
}

TEST_F(perturbation_db_fixture, NoZeroDistanceCandidatesOnCompletedSolution) {
    ASSERT_TRUE(perturbation_db::isValidSolutionInstance(sol));
    EXPECT_TRUE(no_zero_distance_candidates(geo(), sol));
}

// The property must survive the move itself, since perturbate is chained.
TEST_F(perturbation_db_fixture, NoZeroDistanceCandidatesAfterPerturbations) {
    for (int trial = 0; trial < 50; ++trial) {
        p->perturbate();
        ASSERT_TRUE(no_zero_distance_candidates(geo(), sol)) << "after trial " << trial;
    }
}

TEST_F(perturbation_db_fixture, DistancesToReferenceAreZeroOnlyForReference) {
    for (int trial = 0; trial < 50; ++trial) {
        std::vector<int> c = pl->uniformCandidateSample();
        for (int ref : c) {
            std::vector<cost_t> d = pl->getDistancesToReferenceCandidate(c, ref);
            for (std::size_t k = 0; k < c.size(); ++k)
                EXPECT_EQ(d[k] == (cost_t{0, 0}), c[k] == ref)
                    << "candidate " << c[k] << ", ref " << ref;
        }
    }
}

TEST_F(perturbation_db_fixture, ReferenceIsFirstInTopSixForEveryReference) {
    for (int trial = 0; trial < 50; ++trial) {
        std::vector<int> c = pl->uniformCandidateSample();
        for (int ref : c)
            EXPECT_EQ(pl->topSixCandidates(c, ref)[0], ref);
    }
}

TEST_F(perturbation_db_fixture, OrderedIndexesOnKnownDistances) {
    std::vector<cost_t> d = {{5, 0}, {1, 0}, {3, 1}, {3, 0}, {0, 0}};
    EXPECT_EQ(pl->getOrderedIndexes(d), (std::vector<int>{4, 1, 3, 2, 0}));
}

TEST_F(perturbation_db_fixture, OrderedIndexesSortDistances) {
    std::vector<int> c = pl->uniformCandidateSample();
    std::vector<cost_t> d = pl->getDistancesToReferenceCandidate(c, c[0]);
    std::vector<int> o = pl->getOrderedIndexes(d);
    ASSERT_EQ(o.size(), d.size());
    for (std::size_t k = 1; k < o.size(); ++k)
        EXPECT_FALSE(d[o[k]] < d[o[k - 1]]);
}

// ---------------------------------------------------------------------------
// topSixCandidates
// ---------------------------------------------------------------------------

// The six entries are distinct candidates, the reference comes first, and they
// are the six closest candidates to the reference.
TEST_F(perturbation_db_fixture, TopSixAreTheSixClosestCandidates) {
    std::vector<int> c = pl->uniformCandidateSample();
    ASSERT_GE(c.size(), 6u);
    int ref = pl->greedyCandidateSample(c);
    std::array<int, 6> top = pl->topSixCandidates(c, ref);

    EXPECT_EQ(top[0], ref);
    std::set<int> unique(top.begin(), top.end());
    EXPECT_EQ(unique.size(), 6u);

    cost_t farthest_kept{0, 0};
    for (int i : top) {
        ASSERT_NE(std::find(c.begin(), c.end(), i), c.end()) << i;
        cost_t d = i == ref ? cost_t{0, 0}
                            : geo().dist(large_sol.path[ref], large_sol.path[i]);
        if (farthest_kept < d) farthest_kept = d;
    }
    for (int i : c) {
        if (unique.count(i)) continue;
        EXPECT_FALSE(geo().dist(large_sol.path[ref], large_sol.path[i]) < farthest_kept)
            << "candidate " << i << " is closer than a kept one";
    }
}

// ---------------------------------------------------------------------------
// getDoubleBridgeIndexes
// ---------------------------------------------------------------------------

// Four strictly increasing interior indexes of the same parity, so that the
// blocks between them are non-empty and index+1 stays inside the path.
TEST_F(perturbation_db_fixture, DoubleBridgeIndexesAreSortedInteriorSameParity) {
    for (int trial = 0; trial < 20; ++trial) {
        std::array<int, 4> s = pl->getDoubleBridgeIndexes();
        for (int k = 0; k < 4; ++k) {
            EXPECT_GT(s[k], 0);
            EXPECT_LT(s[k], static_cast<int>(large_sol.path.size()) - 1);
            EXPECT_EQ(s[k] % 2, s[0] % 2);
            if (k > 0) EXPECT_LT(s[k - 1], s[k]);
        }
    }
}

// ---------------------------------------------------------------------------
// applyDoubleBridgeMove
// ---------------------------------------------------------------------------

// With cuts {2, 6, 10, 14} on 20 single vertices, the verbatim-copied parts
// must appear in the order A D C B E:
//   A = 0..2, D = 12..14, C = 8..10, B = 4..6, E = 16..19
// (the first vertex of D, C, B, E is reached through geometry.link and may be
//  merged into the linking segment, so it is not checked here).
TEST_F(perturbation_db_fixture, ApplyDoubleBridgeMoveReordersBlocks) {
    std::array<int, 4> cuts = {2, 6, 10, 14};
    pl->applyDoubleBridgeMove(cuts);

    std::vector<int> expected = {0, 1, 2, 12, 13, 14, 8, 9, 10, 4, 5, 6,
                                 16, 17, 18, 19};
    std::size_t j = 0;
    for (auto& s : large_sol.path)
        if (j < expected.size() && s.source == expected[j]) ++j;
    EXPECT_EQ(j, expected.size())
        << "blocks out of order; matched " << j << " of " << expected.size();
}

TEST_F(perturbation_db_fixture, ApplyDoubleBridgeMoveKeepsAllVertices) {
    std::set<int> before = source_set(large_sol);
    std::array<int, 4> cuts = {2, 6, 10, 14};
    pl->applyDoubleBridgeMove(cuts);
    EXPECT_TRUE(keeps_vertices(before, large_sol));
}

TEST_F(perturbation_db_fixture, ApplyDoubleBridgeMoveUpdatesCost) {
    std::array<int, 4> cuts = {2, 6, 10, 14};
    pl->applyDoubleBridgeMove(cuts);
    EXPECT_EQ(large_sol.cost, geo().cost(large_sol));
}

// ---------------------------------------------------------------------------
// d_perturbate
// ---------------------------------------------------------------------------

TEST_F(perturbation_db_fixture, DPerturbateDoesNotThrowOnLargePath) {
    EXPECT_NO_THROW(pl->d_perturbate());
}

TEST_F(perturbation_db_fixture, DPerturbatePreservesAllSourceVertices) {
    std::set<int> before = source_set(large_sol);
    pl->d_perturbate();
    EXPECT_TRUE(keeps_vertices(before, large_sol));
}

// Below 10 segments (isValidSolutionInstance) d_perturbate leaves the path
// unchanged. 3 segments → 1 interior index.
TEST_F(perturbation_db_fixture, DPerturbateTinyPathIsNoOp) {
    cppied_solution small;
    small.path = {{6, 11}, {23, 18}, {44, 42}};
    small.coverage = Eigen::VectorXd::Zero(P.req.size());
    small.cost     = {0, 0};
    perturbation_db_accessor ps(ctx, P, small);
    ps.d_perturbate();
    EXPECT_EQ(small.path.size(), 3u);
}

TEST_F(perturbation_db_fixture, IsValidSolutionInstanceNeedsTenSegments) {
    EXPECT_FALSE(perturbation_db::isValidSolutionInstance(single_vertex_path(9)));
    EXPECT_TRUE(perturbation_db::isValidSolutionInstance(single_vertex_path(10)));
}

// 9 segments: below the guard, the path is left untouched.
TEST_F(perturbation_db_fixture, DPerturbateBelowTenSegmentsIsNoOp) {
    cppied_solution s = single_vertex_path(9);
    perturbation_db_accessor q(ctx, P, s);
    std::vector<segment> before = s.path;
    q.d_perturbate();
    ASSERT_EQ(s.path.size(), before.size());
    for (std::size_t k = 0; k < before.size(); ++k)
        EXPECT_EQ(s.path[k].source, before[k].source);
}

// 10–13 segments: only 4 or 5 candidates for one parity (< 6), which must
// still give a valid double bridge.
TEST_F(perturbation_db_fixture, DPerturbateWithFewCandidatesKeepsVertices) {
    for (int n : {10, 11, 12, 13}) {
        for (int trial = 0; trial < 20; ++trial) {
            cppied_solution s = single_vertex_path(n);
            perturbation_db_accessor q(ctx, P, s);
            std::set<int> before = source_set(s);
            std::array<int, 4> idx = q.getDoubleBridgeIndexes();
            for (int k = 0; k < 4; ++k) {
                EXPECT_GE(idx[k], 1) << "n=" << n;
                EXPECT_LT(idx[k], n - 1) << "n=" << n;
                if (k > 0) EXPECT_LT(idx[k - 1], idx[k]) << "n=" << n;
            }
            q.d_perturbate();
            EXPECT_TRUE(keeps_vertices(before, s)) << "n=" << n;
        }
    }
}

// ---------------------------------------------------------------------------
// perturbate (public interface: d_perturbate + complete + reset)
// ---------------------------------------------------------------------------

TEST_F(perturbation_db_fixture, PerturbateDoesNotThrow) {
    EXPECT_NO_THROW(pl->perturbate());
}

TEST_F(perturbation_db_fixture, PerturbateCostComputableAfterComplete) {
    pl->perturbate();
    EXPECT_NO_THROW({ large_sol.cost = geo().cost(large_sol); });
}

TEST_F(perturbation_db_fixture, PerturbateDoesNotThrowOnSmallSolution) {
    EXPECT_NO_THROW(p->perturbate());
}

// The double bridge only reorders the path: a feasible solution stays feasible.
TEST_F(perturbation_db_fixture, PerturbateKeepsCoverageSatisfied) {
    ASSERT_TRUE(cov().satisfy(sol));
    for (int trial = 0; trial < 10; ++trial) {
        p->perturbate();
        EXPECT_TRUE(cov().satisfy(sol)) << "trial " << trial;
    }
}

// ============================================================================
// Edge cases and remaining functions
// ============================================================================

// ---------------------------------------------------------------------------
// perturbation (base class)
// ---------------------------------------------------------------------------

// Minimal concrete perturbation: d_perturbate does nothing, so perturbate()
// shows exactly what the base class adds (complete + reset).
struct noop_perturbation : public perturbation {
    using perturbation::perturbation;
    int calls = 0;
    void d_perturbate() override { ++calls; }
    using perturbation::geometry;
    using perturbation::coverage;
};

TEST_F(perturbation_base_fixture, BaseUnsupportedInterfaceThrows) {
    noop_perturbation q(ctx, P, sol);
    EXPECT_THROW(q.initialize(), std::runtime_error);
    EXPECT_THROW(q.terminate(), std::runtime_error);
    EXPECT_THROW(q.d_solve(sol), std::runtime_error);
}

TEST_F(perturbation_base_fixture, BasePerturbateCallsDPerturbateOnce) {
    noop_perturbation q(ctx, P, sol);
    q.perturbate();
    EXPECT_EQ(q.calls, 1);
}

// perturbate = d_perturbate + complete + reset: on the uncompleted raw_sol it
// must give the completed path and coverage built in the fixture.
TEST_F(perturbation_base_fixture, BasePerturbateCompletesAndResetsCoverage) {
    cppied_solution s = raw_sol;
    s.coverage.setZero();
    noop_perturbation q(ctx, P, s);
    q.perturbate();
    ASSERT_EQ(s.path.size(), sol.path.size());
    for (std::size_t k = 0; k < s.path.size(); ++k) {
        EXPECT_EQ(s.path[k].source, sol.path[k].source) << "segment " << k;
        EXPECT_EQ(s.path[k].target, sol.path[k].target) << "segment " << k;
    }
    EXPECT_TRUE(s.coverage.isApprox(sol.coverage, 1e-9));
}

TEST_F(perturbation_base_fixture, BasePerturbateOnEmptyPathIsNoOp) {
    cppied_solution s;
    s.coverage = Eigen::VectorXd::Zero(P.req.size());
    s.cost = {0, 0};
    noop_perturbation q(ctx, P, s);
    EXPECT_NO_THROW(q.perturbate());
    EXPECT_TRUE(s.path.empty());
    EXPECT_EQ(s.coverage.sum(), 0.0);
}

// for_each_segment visits the horizontal tracks then the vertical tracks of
// the box, exactly as the PT replacement helpers build them.
TEST_F(perturbation_base_fixture, ForEachSegmentMatchesCoverReplacements) {
    noop_perturbation q(ctx, P, sol);
    perturbation_pt_accessor pt(ctx, P, sol);
    iRectangle box;
    box.ul = {1, 2};
    box.lr = {3, 4};
    std::vector<segment> visited;
    q.for_each_segment(box, [&](const segment& s){ visited.push_back(s); });

    std::vector<segment> expected = pt.getHorizontalCoverReplacements(box);
    std::vector<segment> v = pt.getVerticalCoverReplacements(box);
    expected.insert(expected.end(), v.begin(), v.end());
    ASSERT_EQ(visited.size(), expected.size());
    for (std::size_t k = 0; k < visited.size(); ++k) {
        EXPECT_EQ(visited[k].source, expected[k].source);
        EXPECT_EQ(visited[k].target, expected[k].target);
    }
}

TEST_F(perturbation_base_fixture, ForEachSegmentOnEmptyBoxVisitsNothing) {
    noop_perturbation q(ctx, P, sol);
    iRectangle box;  // default: ul = INT_MAX, lr = INT_MIN
    int n = 0;
    q.for_each_segment(box, [&](const segment&){ ++n; });
    EXPECT_EQ(n, 0);
}

// ---------------------------------------------------------------------------
// perturbation_ri — edge cases
// ---------------------------------------------------------------------------

// Empty path → sampleSize 0 → nothing is sampled nor inserted.
TEST_F(perturbation_base_fixture, RiOnEmptyPathIsNoOp) {
    cppied_solution s;
    s.coverage = Eigen::VectorXd::Zero(P.req.size());
    s.cost = {0, 0};
    perturbation_ri_accessor q(ctx, P, s);
    EXPECT_EQ(q.sampleSize, 0u);
    EXPECT_NO_THROW(q.d_perturbate());
    EXPECT_TRUE(s.path.empty());
}

// One segment → sampleSize ceil(0.05) = 1 → one insertion.
TEST_F(perturbation_base_fixture, RiOnSingleSegmentPathInsertsOne) {
    cppied_solution s;
    s.path = {sol.path.front()};
    s.coverage = Eigen::VectorXd::Zero(P.req.size());
    perturbation_ri_accessor q(ctx, P, s);
    q.coverage.reset(s);
    s.cost = q.geometry.cost(s);
    ASSERT_EQ(q.sampleSize, 1u);
    q.d_perturbate();
    EXPECT_EQ(s.path.size(), 2u);
    EXPECT_EQ(s.cost, q.geometry.cost(s));
}

// sampleSize == 0 and sampleSize == #vertices are the two nth_element bounds.
TEST_F(perturbation_ri_fixture, SortedCandidatesAtSampleSizeBounds) {
    std::vector<double> w = p->getCandidateWeights();
    for (std::size_t k : {std::size_t{0}, w.size()}) {
        p->sampleSize = k;
        std::vector<int> c = p->getSortedCandidates(w);
        std::vector<int> sorted = c;
        std::sort(sorted.begin(), sorted.end());
        for (int v = 0; v < static_cast<int>(sorted.size()); ++v)
            ASSERT_EQ(sorted[v], v) << "sampleSize " << k;
    }
}

TEST_F(perturbation_ri_fixture, SetCandidateSamplingWithZeroSampleSize) {
    p->sampleSize = 0;
    p->insertionSample.clear();
    p->setCandidateSampling({3, 1, 2});
    EXPECT_TRUE(p->insertionSample.empty());
}

// The inserted segment is a single-vertex segment on the requested vertex.
TEST_F(perturbation_ri_fixture, InsertCandidateInsertsRequestedVertex) {
    for (int v : {0, 17, static_cast<int>(P.vertex.size()) - 1}) {
        auto count = [&]{
            return std::count_if(sol.path.begin(), sol.path.end(),
                [&](const segment& s){
                    return s.source == v && !path_engine::is_node(s.target); });
        };
        auto before = count();
        segment s{v, path_engine::NULL_NODE};
        p->insertCandidate(s);
        EXPECT_EQ(count(), before + 1) << "vertex " << v;
    }
}

// The incremental insertion cost must match the recomputed path cost.
TEST_F(perturbation_ri_fixture, InsertCandidateKeepsCostConsistent) {
    ASSERT_EQ(sol.cost, geo().cost(sol));
    segment s{0, path_engine::NULL_NODE};
    p->insertCandidate(s);
    EXPECT_EQ(sol.cost, geo().cost(sol));
}

TEST_F(perturbation_ri_fixture, DPerturbateKeepsCostAndCoverageConsistent) {
    p->d_perturbate();
    EXPECT_EQ(sol.cost, geo().cost(sol));
    cppied_solution check = sol;
    cov().reset(check);
    EXPECT_TRUE(sol.coverage.isApprox(check.coverage, 1e-9));
}

// RI only adds segments: a feasible solution stays feasible.
TEST_F(perturbation_ri_fixture, PerturbateKeepsCoverageSatisfied) {
    ASSERT_TRUE(cov().satisfy(sol));
    p->perturbate();
    EXPECT_TRUE(cov().satisfy(sol));
}

// ILS compares trial.cost right after perturbate: it must not be stale.
TEST_F(perturbation_ri_fixture, PerturbateLeavesUpToDateCost) {
    p->perturbate();
    EXPECT_EQ(sol.cost, geo().cost(sol));
}

// ---------------------------------------------------------------------------
// perturbation_pt — edge cases
// ---------------------------------------------------------------------------

TEST_F(perturbation_pt_fixture, SampleCandidatesWithZeroKIsEmpty) {
    p->K = 0;
    EXPECT_TRUE(p->sampleCandidates().empty());
}

// K = size-1 samples every index but 0, including the last one.
TEST_F(perturbation_pt_fixture, SampleCandidatesWithMaximalKIsEveryIndex) {
    p->K = sol.path.size() - 1;
    std::vector<int> c = p->sampleCandidates();
    ASSERT_EQ(c.size(), sol.path.size() - 1);
    for (int k = 0; k < static_cast<int>(c.size()); ++k)
        EXPECT_EQ(c[k], k + 1);
}

TEST_F(perturbation_pt_fixture, CoverReplacementsOfEmptyBoxAreEmpty) {
    iRectangle box;  // what unsatisfied_rectangle gives for no cell
    EXPECT_TRUE(p->getHorizontalCoverReplacements(box).empty());
    EXPECT_TRUE(p->getVerticalCoverReplacements(box).empty());
}

// Boxes touching the grid border: tracks are clamped to valid vertices and
// keep their orientation.
TEST_F(perturbation_pt_fixture, CoverReplacementsAtGridCornersAreValidVertices) {
    const int R = geo().n_rows, C = geo().n_cols, HB = geo().horizontal_bound;
    const int NV = static_cast<int>(P.vertex.size());
    std::vector<std::pair<iCoordinate, iCoordinate>> boxes = {
        {{0, 0}, {1, 1}}, {{C - 1, R - 1}, {C, R}},
        {{0, R - 1}, {C, R}}, {{0, 0}, {C, R}}};
    for (auto& [ul, lr] : boxes) {
        iRectangle box;
        box.ul = ul;
        box.lr = lr;
        auto h = p->getHorizontalCoverReplacements(box);
        auto v = p->getVerticalCoverReplacements(box);
        EXPECT_FALSE(h.empty());
        EXPECT_FALSE(v.empty());
        for (auto& s : h) {
            EXPECT_GE(s.source, 0);
            EXPECT_LT(s.source, HB);
            if (path_engine::is_node(s.target)) EXPECT_LT(s.target, HB);
            EXPECT_TRUE(geo().is_horizontal(s));
        }
        for (auto& s : v) {
            EXPECT_GE(s.source, HB);
            EXPECT_LT(s.source, NV);
            if (path_engine::is_node(s.target)) EXPECT_LT(s.target, NV);
            EXPECT_FALSE(geo().is_horizontal(s));
        }
    }
}

// A 1-cell-wide box gives single-vertex tracks (target NULL_NODE).
TEST_F(perturbation_pt_fixture, CoverReplacementsOfUnitBoxAreSingleVertex) {
    iRectangle box;
    box.ul = {2, 2};
    box.lr = {3, 3};
    for (auto& s : p->getHorizontalCoverReplacements(box))
        EXPECT_EQ(s.target, path_engine::NULL_NODE);
    for (auto& s : p->getVerticalCoverReplacements(box))
        EXPECT_EQ(s.target, path_engine::NULL_NODE);
}

// Every segment of sol, wherever it lies in the grid: its replacements cover
// all the cells its removal leaves unsatisfied.
TEST_F(perturbation_pt_fixture, ReplacementsCoverUnsatisfiedCellsForEverySegment) {
    for (int i = 0; i < static_cast<int>(sol.path.size()); ++i) {
        cppied_solution s = sol;
        perturbation_pt_accessor q(ctx, P, s);
        q.coverage.remove(s, s.path[i]);
        std::vector<int> unsat;
        q.coverage.unsatisfied_cells(s, s.path[i], unsat);
        q.setReplacementsCoverage(q.getReplacements(s.path[i]));
        for (int c : unsat)
            EXPECT_GE(s.coverage(c), P.req(c)) << "segment " << i << ", cell " << c;
    }
}

// When both orientations of the first track cost the same, both directions
// must come out (random tie-break).
TEST_F(perturbation_pt_fixture, ReplacementDirectionTieBreaksBothWays) {
    // Every horizontal and vertical track of the grid, of any length.
    std::vector<segment> tracks;
    for (int row = 0; row <= geo().n_rows; ++row)
        for (int a = 0; a < geo().n_cols; ++a)
            for (int b = a; b < geo().n_cols; ++b) {
                int u = row * geo().n_cols;
                tracks.push_back({u + a, a == b ? path_engine::NULL_NODE : u + b});
            }
    for (int col = 0; col <= geo().n_cols; ++col)
        for (int a = 0; a < geo().n_rows; ++a)
            for (int b = a; b < geo().n_rows; ++b) {
                int u = geo().horizontal_bound + col * geo().n_rows;
                tracks.push_back({u + a, a == b ? path_engine::NULL_NODE : u + b});
            }

    for (const segment& back : sol.path)
        for (const segment& t : tracks) {
            if (geo().dist(back, t) != geo().dist(back, geo().flip_segment(t)))
                continue;
            std::vector<segment> new_sol = {back};
            std::set<direction> seen;
            for (int trial = 0; trial < 200; ++trial)
                seen.insert(p->getReplacementDirection(new_sol, {t}));
            EXPECT_EQ(seen.size(), 2u) << "track " << t.source << "-" << t.target;
            return;
        }
    GTEST_SKIP() << "no tied track found on this instance";
}

// With a strictly cheaper orientation, it is always chosen.
TEST_F(perturbation_pt_fixture, ReplacementDirectionStrictlyCheaperIsDeterministic) {
    std::vector<segment> new_sol = {sol.path.front()};
    const segment& back = new_sol.back();
    for (int row = 0; row <= geo().n_rows; ++row)
        for (int x = 0; x + 1 < geo().n_cols; ++x) {
            segment t{row * geo().n_cols + x, row * geo().n_cols + x + 1};
            segment f = geo().flip_segment(t);
            cost_t ct = geo().dist(back, t), cf = geo().dist(back, f);
            if (ct == cf) continue;
            direction best = geo().get_direction(cf < ct ? f : t);
            for (int trial = 0; trial < 50; ++trial)
                ASSERT_EQ(p->getReplacementDirection(new_sol, {t}), best);
            return;
        }
    GTEST_SKIP() << "no track with distinct orientation costs";
}

TEST_F(perturbation_pt_fixture, PushBackParallelTracksWithNoTrackIsNoOp) {
    std::vector<segment> new_sol = {sol.path[0]};
    p->pushBackParallelTracks(new_sol, {}, direction::E);
    EXPECT_EQ(new_sol.size(), 1u);
}

// Vertical tracks alternate S/N.
TEST_F(perturbation_pt_fixture, PushBackParallelVerticalTracksAlternate) {
    iRectangle box;
    box.ul = {1, 2};
    box.lr = {3, 5};
    std::vector<segment> r = p->getVerticalCoverReplacements(box);
    ASSERT_GE(r.size(), 2u);
    std::vector<segment> new_sol;
    p->pushBackParallelTracks(new_sol, r, direction::N);
    for (std::size_t k = 0; k < new_sol.size(); ++k)
        EXPECT_EQ(geo().get_direction(new_sol[k]),
                  k % 2 == 0 ? direction::N : direction::S) << "track " << k;
}

// Replacing the last segment of the path.
TEST_F(perturbation_pt_fixture, ReplaceCandidatesLastIndexKeepsFeasibility) {
    p->replaceCandidates({static_cast<int>(sol.path.size()) - 1});
    EXPECT_TRUE(coverage_consistent(sol));
    EXPECT_TRUE(cov().satisfy(sol));
}

// Two adjacent indices: the second one still refers to the original path.
TEST_F(perturbation_pt_fixture, ReplaceCandidatesAdjacentIndicesKeepFeasibility) {
    p->replaceCandidates({1, 2});
    EXPECT_TRUE(coverage_consistent(sol));
    EXPECT_TRUE(cov().satisfy(sol));
}

// Every segment but the first replaced: path[0] stays in front and the cells
// stay covered.
TEST_F(perturbation_pt_fixture, ReplaceCandidatesAllIndicesKeepFeasibility) {
    segment first = sol.path.front();
    std::vector<int> all(sol.path.size() - 1);
    std::iota(all.begin(), all.end(), 1);
    p->replaceCandidates(all);
    ASSERT_FALSE(sol.path.empty());
    EXPECT_EQ(sol.path.front().source, first.source);
    EXPECT_EQ(sol.path.front().target, first.target);
    EXPECT_TRUE(coverage_consistent(sol));
    EXPECT_TRUE(cov().satisfy(sol));
}

TEST_F(perturbation_pt_fixture, DPerturbateKeepsCoverageSatisfied) {
    for (int trial = 0; trial < 10; ++trial) {
        cppied_solution s = sol;
        perturbation_pt_accessor q(ctx, P, s);
        q.d_perturbate();
        EXPECT_TRUE(q.coverage.satisfy(s)) << "trial " << trial;
        EXPECT_TRUE(coverage_consistent(s)) << "trial " << trial;
    }
}

TEST_F(perturbation_pt_fixture, DPerturbateLeavesUpToDateCost) {
    p->d_perturbate();
    EXPECT_EQ(sol.cost, geo().cost(sol));
}

TEST_F(perturbation_pt_fixture, PerturbateLeavesUpToDateCost) {
    p->perturbate();
    EXPECT_EQ(sol.cost, geo().cost(sol));
}

// Smallest path with a candidate: 2 segments, K = 1, only index 1.
TEST_F(perturbation_base_fixture, PtOnTwoSegmentPath) {
    cppied_solution s;
    s.path = {raw_sol.path[0], raw_sol.path[1]};
    s.coverage = Eigen::VectorXd::Zero(P.req.size());
    perturbation_pt_accessor q(ctx, P, s);
    q.coverage.reset(s);
    s.cost = q.geometry.cost(s);
    ASSERT_EQ(q.K, 1u);
    EXPECT_EQ(q.sampleCandidates(), (std::vector<int>{1}));
    EXPECT_NO_THROW(q.perturbate());
    EXPECT_FALSE(s.path.empty());
}

// ---------------------------------------------------------------------------
// perturbation_db — edge cases
// ---------------------------------------------------------------------------

TEST_F(perturbation_db_fixture, IsValidSolutionInstanceRejectsEmptyPath) {
    cppied_solution s;
    EXPECT_FALSE(perturbation_db::isValidSolutionInstance(s));
}

// With a single sampled index, no other index to join: gain keeps its
// sentinel and the greedy choice is that index.
TEST_F(perturbation_db_fixture, GainAndGreedyWithSingleCandidate) {
    std::vector<int> c = {5};
    cost_t g = pl->gain(c, 5);
    EXPECT_EQ(g.length, std::numeric_limits<int>::max());
    EXPECT_EQ(g.turns, std::numeric_limits<int>::max());
    EXPECT_EQ(pl->greedyCandidateSample(c), 5);
}

TEST_F(perturbation_db_fixture, OrderedIndexesOfEmptyIsEmpty) {
    std::vector<cost_t> d;
    EXPECT_TRUE(pl->getOrderedIndexes(d).empty());
}

// Fewer than 6 candidates: the first entries are the candidates (ref first),
// the rest is zero-padded.
TEST_F(perturbation_db_fixture, TopSixWithFourCandidatesIsZeroPadded) {
    cppied_solution s = single_vertex_path(10);
    perturbation_db_accessor q(ctx, P, s);
    std::vector<int> c = {1, 3, 5, 7};
    std::array<int, 6> t = q.topSixCandidates(c, 5);
    EXPECT_EQ(t[0], 5);
    EXPECT_EQ(std::set<int>(t.begin(), t.begin() + 4), (std::set<int>{1, 3, 5, 7}));
    EXPECT_EQ(t[4], 0);
    EXPECT_EQ(t[5], 0);
}

// Tightest cuts on the smallest valid path, including the last cut right
// before the final segment.
TEST_F(perturbation_db_fixture, ApplyDoubleBridgeMoveTightestCuts) {
    for (std::array<int, 4> cuts : {std::array<int, 4>{1, 3, 5, 7},
                                    std::array<int, 4>{2, 4, 6, 8}}) {
        cppied_solution s = single_vertex_path(10);
        perturbation_db_accessor q(ctx, P, s);
        std::set<int> before = source_set(s);
        q.applyDoubleBridgeMove(cuts);
        EXPECT_TRUE(keeps_vertices(before, s)) << "cuts " << cuts[0];
        EXPECT_EQ(s.cost, q.geometry.cost(s)) << "cuts " << cuts[0];
        EXPECT_EQ(s.path.front().source, 0);
    }
}

// On a real (completed) solution the move keeps every visited vertex, so the
// coverage after reset is still satisfied.
TEST_F(perturbation_db_fixture, ApplyDoubleBridgeMoveOnCompletedSolution) {
    ASSERT_GE(sol.path.size(), 12u);
    std::set<int> before = node_set(sol);
    std::array<int, 4> cuts = {1, 5, 7, 11};
    p->applyDoubleBridgeMove(cuts);
    EXPECT_TRUE(keeps_vertices(before, sol));
    cov().reset(sol);
    EXPECT_TRUE(cov().satisfy(sol));
}

TEST_F(perturbation_db_fixture, PerturbateLeavesUpToDateCost) {
    p->perturbate();
    EXPECT_EQ(sol.cost, geo().cost(sol));
}
