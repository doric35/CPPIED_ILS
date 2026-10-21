#include "../test_fixture.hpp"
#include "../../include/algorithms/MinimumPrice.hpp"

// ============================================================================
// Tests of the pricing subproblem solver MinimumPrice (one reallocated index R).
//
// The dynamic programs are checked against a brute force over every segment of
// a line. A forward path f -> t (f <= t) is priced
//     source(f) + sum_{l=f+1..t} intern(l) + target(t)
// and a backward path f -> t (f >= t)
//     source(f) + sum_{l=t..f-1} intern(l) + target(t).
// The phase solvers are checked against the LP reduced cost of the column of a
// segment S at index R:
//     phase 1: gain(S).length - sum_{w in S} vertexPrices(w) - reallocationPrice
//     phase 2: gain(S).turns  - sum_{w in S} vertexPrices(w) - reallocationPrice
//              - lengthPrice * gain(S).length
// where gain(S) is the insertion cost of S between path[R-1] and path[R+1]
// (tail insertion when R is the last index).
// ============================================================================

namespace {

struct test_prices {
    std::vector<double> source, intern, target;
};

test_prices makePrices(int n_vertex, int seed) {
    test_prices p{std::vector<double>(n_vertex), std::vector<double>(n_vertex), std::vector<double>(n_vertex)};
    for (int l = 0; l < n_vertex; ++l) {
        p.source[l] = ((l * 37 + seed * 11) % 11) - 5.0 + 1e-3 * l;
        p.intern[l] = ((l * 53 + seed * 7)  % 7)  - 3.5 + 1e-4 * l;
        p.target[l] = ((l * 29 + seed * 5)  % 13) - 6.0 + 1e-5 * l;
    }
    return p;
}

test_prices constantInternPrices(int n_vertex, double intern) {
    test_prices p = makePrices(n_vertex, 0);
    std::fill(p.intern.begin(), p.intern.end(), intern);
    return p;
}

double forwardValue(const test_prices& p, int f, int t) {
    double v = p.source[f];
    for (int l = f + 1; l <= t; ++l) v += p.intern[l];
    return v + p.target[t];
}

double backwardValue(const test_prices& p, int f, int t) {
    double v = p.source[f];
    for (int l = t; l <= f - 1; ++l) v += p.intern[l];
    return v + p.target[t];
}

double bruteForward(const test_prices& p, int u, int v) {
    double best = std::numeric_limits<double>::max();
    for (int f = u; f <= v; ++f)
        for (int t = f; t <= v; ++t) best = std::min(best, forwardValue(p, f, t));
    return best;
}

double bruteBackward(const test_prices& p, int u, int v) {
    double best = std::numeric_limits<double>::max();
    for (int f = u; f <= v; ++f)
        for (int t = u; t <= f; ++t) best = std::min(best, backwardValue(p, f, t));
    return best;
}

// First and last vertex (inclusive) of every horizontal and vertical line.
std::vector<std::pair<int, int>> allLines(const path_engine& g) {
    std::vector<std::pair<int, int>> lines;
    for (int i = 0; i < g.horizontal_bound; i += g.n_cols) lines.emplace_back(i, i + g.n_cols - 1);
    for (int j = g.horizontal_bound; j < g.n_vertex; j += g.n_rows) lines.emplace_back(j, j + g.n_rows - 1);
    return lines;
}

// Endpoints of an oriented segment, single vertices included.
std::pair<int, int> endpoints(const segment& s) {
    return {s.source, path_engine::is_node(s.target) ? s.target : s.source};
}

// MinimumPrice with per-vertex test prices; exposes the protected steps.
class TestMinimumPrice : public MinimumPrice {
public:
    template<typename... Args>
    explicit TestMinimumPrice(test_prices prices, Args&&... args) :
        MinimumPrice(std::forward<Args>(args)...), p(std::move(prices)) {}

    using MinimumPrice::getForwardDp;
    using MinimumPrice::getBackwardDp;
    using MinimumPrice::getReallocation;
    using MinimumPrice::solveEdge;
    using MinimumPrice::solveHorizontal;
    using MinimumPrice::solveVertical;
    using MinimumPrice::updateIncumbent;
    using MinimumPrice::incumbent;

protected:
    double sourceLinkPrice(const segment& s) override { return p.source[s.source]; }
    double internLinkPrice(const segment& s) override { return p.intern[s.source]; }
    double targetLinkPrice(const segment& s) override { return p.target[s.source]; }

private:
    test_prices p;
};

} // namespace

// ============================================================================
// Fixture
// ============================================================================

class minimum_price_fixture : public cppied_context_fixture {
protected:
    path_engine g{P};
    coverage_engine c{P};
    cppied_solution sol;
    Eigen::VectorXd vertexPrices;

    void SetUp() override {
        sol.path = {{6,  11},
                    {23, 18},
                    {44, 42},
                    {48, 51},
                    {30, 35},
                    {75, 73},
                    {62, 63}};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        c.reset(sol);
        sol.cost = g.cost(sol);
        vertexPrices = Eigen::VectorXd::Zero(g.n_vertex);
        ctx.start_time = std::chrono::high_resolution_clock::now();
    }

    TestMinimumPrice makeSolver(const test_prices& p, int R = 3) {
        return TestMinimumPrice(p, vertexPrices, 0.0, 0.0, P, c, g, sol, R);
    }

    // Non-negative coverage prices, as given by the duals of >= rows.
    void setVertexPrices(int seed) {
        for (int w = 0; w < g.n_vertex; ++w) vertexPrices(w) = 0.05 * ((w * 17 + seed * 3) % 9);
    }

    cost_t gain(int R, const segment& s) {
        if (R == (int)sol.path.size() - 1) return g.insert_tail_gain(sol.path[R - 1], s);
        return g.insert_between_gain(sol.path[R - 1], s, sol.path[R + 1]);
    }

    double coveragePrice(const segment& s) {
        auto [a, b] = endpoints(s);
        double sum = 0.0;
        for (int w = std::min(a, b); w <= std::max(a, b); ++w) sum += vertexPrices(w);
        return sum;
    }

    double firstStageReducedCost(int R, const segment& s, double reallocationPrice) {
        return gain(R, s).length - coveragePrice(s) - reallocationPrice;
    }

    double secondStageReducedCost(int R, const segment& s, double reallocationPrice, double lengthPrice) {
        cost_t gs = gain(R, s);
        return gs.turns - coveragePrice(s) - reallocationPrice - lengthPrice * gs.length;
    }

    // Every segment of the grid, both orientations, single vertices included.
    std::vector<segment> allSegments() {
        std::vector<segment> segments;
        for (auto [u, v] : allLines(g))
            for (int f = u; f <= v; ++f) {
                segments.push_back({f, path_engine::NULL_NODE});
                segments.push_back({f, path_engine::REVERSED_NULL_NODE});
                for (int t = f + 1; t <= v; ++t) {
                    segments.push_back({f, t});
                    segments.push_back({t, f});
                }
            }
        return segments;
    }
};

// ============================================================================
// Dynamic programs
// ============================================================================

TEST_F(minimum_price_fixture, ForwardDpMatchesBruteForce) {
    for (int seed = 0; seed < 4; ++seed) {
        test_prices p = makePrices(g.n_vertex, seed);
        TestMinimumPrice solver = makeSolver(p);
        for (auto [u, v] : allLines(g)) {
            Reallocation r = solver.getReallocation(solver.getForwardDp(u, v));
            int f = u + r.S.source, t = u + r.S.target;    // DP indices are relative to u
            ASSERT_GE(f, u); ASSERT_LE(t, v); ASSERT_LE(f, t) << "forward path must go u -> v";
            EXPECT_NEAR(r.value, bruteForward(p, u, v), 1e-9) << "line [" << u << "," << v << "], seed " << seed;
            EXPECT_NEAR(forwardValue(p, f, t), r.value, 1e-9) << "value inconsistent with returned segment";
            EXPECT_EQ(r.R, 3);
        }
    }
}

TEST_F(minimum_price_fixture, BackwardDpMatchesBruteForce) {
    for (int seed = 0; seed < 4; ++seed) {
        test_prices p = makePrices(g.n_vertex, seed);
        TestMinimumPrice solver = makeSolver(p);
        for (auto [u, v] : allLines(g)) {
            Reallocation r = solver.getReallocation(solver.getBackwardDp(u, v));
            int f = v - r.S.source, t = v - r.S.target;    // DP indices are relative to v
            ASSERT_LE(f, v); ASSERT_GE(t, u); ASSERT_GE(f, t) << "backward path must go v -> u";
            EXPECT_NEAR(r.value, bruteBackward(p, u, v), 1e-9) << "line [" << u << "," << v << "], seed " << seed;
            EXPECT_NEAR(backwardValue(p, f, t), r.value, 1e-9) << "value inconsistent with returned segment";
        }
    }
}

// ============================================================================
// solveEdge — best orientation on one line, absolute vertex ids
// ============================================================================

TEST_F(minimum_price_fixture, SolveEdgeTakesBestOrientation) {
    for (int seed = 0; seed < 4; ++seed) {
        test_prices p = makePrices(g.n_vertex, seed);
        TestMinimumPrice solver = makeSolver(p);
        for (auto [u, v] : allLines(g)) {
            const double fwd = bruteForward(p, u, v), bwd = bruteBackward(p, u, v);
            Reallocation r = solver.solveEdge(u, v);
            auto [f, t] = endpoints(r.S);
            ASSERT_TRUE(f >= u && f <= v && t >= u && t <= v)
                << "segment {" << r.S.source << "," << r.S.target << "} outside line [" << u << "," << v << "]";
            EXPECT_NEAR(r.value, std::min(fwd, bwd), 1e-9) << "line [" << u << "," << v << "], seed " << seed;

            const bool backward = r.S.target == path_engine::REVERSED_NULL_NODE ||
                                  (path_engine::is_node(r.S.target) && r.S.target < r.S.source);
            EXPECT_NEAR(backward ? backwardValue(p, f, t) : forwardValue(p, f, t), r.value, 1e-9)
                << "value inconsistent with segment {" << r.S.source << "," << r.S.target << "}";
            if (f == t)
                EXPECT_TRUE(r.S.target == path_engine::NULL_NODE || r.S.target == path_engine::REVERSED_NULL_NODE);
        }
    }
}

// Very expensive internal links: the optimum is a single vertex.
TEST_F(minimum_price_fixture, SolveEdgeReturnsSingleVertexWhenLinksAreExpensive) {
    TestMinimumPrice solver = makeSolver(constantInternPrices(g.n_vertex, 100.0));
    Reallocation r = solver.solveEdge(0, g.n_cols - 1);
    EXPECT_FALSE(path_engine::is_node(r.S.target)) << "expected a single vertex";
    EXPECT_TRUE(path_engine::is_node(r.S.source));
}

// Very cheap internal links: the optimum spans the whole line.
TEST_F(minimum_price_fixture, SolveEdgeReturnsWholeLineWhenLinksAreCheap) {
    TestMinimumPrice solver = makeSolver(constantInternPrices(g.n_vertex, -100.0));
    const int u = 0, v = g.n_cols - 1;
    Reallocation r = solver.solveEdge(u, v);
    EXPECT_TRUE(r.S == (segment{u, v}) || r.S == (segment{v, u}))
        << "got {" << r.S.source << "," << r.S.target << "}";
}

// ============================================================================
// updateIncumbent
// ============================================================================

TEST_F(minimum_price_fixture, UpdateIncumbentKeepsMinimumValue) {
    TestMinimumPrice solver = makeSolver(makePrices(g.n_vertex, 0));
    EXPECT_EQ(solver.incumbent.value, MAX_INT) << "the incumbent must start at +infinity";

    solver.updateIncumbent({{1, 2}, -1.0, 3, {0, 0}});
    EXPECT_EQ(solver.incumbent.S, (segment{1, 2}));
    solver.updateIncumbent({{4, 5}, -0.5, 3, {0, 0}});
    EXPECT_EQ(solver.incumbent.S, (segment{1, 2})) << "a worse candidate replaced the incumbent";
    solver.updateIncumbent({{4, 5}, -1.0, 3, {0, 0}});
    EXPECT_EQ(solver.incumbent.S, (segment{1, 2})) << "an equal candidate replaced the incumbent";
    solver.updateIncumbent({{7, 8}, -2.0, 3, {0, 0}});
    EXPECT_EQ(solver.incumbent.S, (segment{7, 8}));
}

// ============================================================================
// solveHorizontal / solveVertical / solve — best line of the grid
// ============================================================================

TEST_F(minimum_price_fixture, SolveHorizontalTakesBestHorizontalLine) {
    test_prices p = makePrices(g.n_vertex, 1);
    TestMinimumPrice solver = makeSolver(p);
    double best = std::numeric_limits<double>::max();
    for (int u = 0; u < g.horizontal_bound; u += g.n_cols) {
        const int v = u + g.n_cols - 1;
        best = std::min({best, bruteForward(p, u, v), bruteBackward(p, u, v)});
    }
    solver.solveHorizontal();
    EXPECT_NEAR(solver.incumbent.value, best, 1e-9);
    EXPECT_LT(solver.incumbent.S.source, g.horizontal_bound);
}

TEST_F(minimum_price_fixture, SolveVerticalTakesBestVerticalLine) {
    test_prices p = makePrices(g.n_vertex, 1);
    TestMinimumPrice solver = makeSolver(p);
    double best = std::numeric_limits<double>::max();
    for (int u = g.horizontal_bound; u < g.n_vertex; u += g.n_rows) {
        const int v = u + g.n_rows - 1;
        best = std::min({best, bruteForward(p, u, v), bruteBackward(p, u, v)});
    }
    solver.solveVertical();
    EXPECT_NEAR(solver.incumbent.value, best, 1e-9);
    EXPECT_GE(solver.incumbent.S.source, g.horizontal_bound);
    EXPECT_LT(solver.incumbent.S.source, g.n_vertex);
}

TEST_F(minimum_price_fixture, SolveTakesBestLineOfTheGrid) {
    for (int seed = 0; seed < 4; ++seed) {
        test_prices p = makePrices(g.n_vertex, seed);
        double best = std::numeric_limits<double>::max();
        for (auto [u, v] : allLines(g)) best = std::min({best, bruteForward(p, u, v), bruteBackward(p, u, v)});

        Reallocation r = makeSolver(p, 2).solve();
        EXPECT_NEAR(r.value, best, 1e-9) << "seed " << seed;
        EXPECT_EQ(r.R, 2);
        auto [f, t] = endpoints(r.S);
        ASSERT_TRUE(path_engine::is_node(f));
        EXPECT_LT(std::max(f, t), g.n_vertex);
        const bool same_line = std::any_of(allLines(g).begin(), allLines(g).end(),
            [&](auto line) { return std::min(f, t) >= line.first && std::max(f, t) <= line.second; });
        EXPECT_TRUE(same_line) << "segment {" << r.S.source << "," << r.S.target << "} spans two lines";
    }
}

// ============================================================================
// Phase solvers — the returned value is the minimum LP reduced cost
// ============================================================================

TEST_F(minimum_price_fixture, FirstStageReturnsMinimumReducedCost) {
    const double reallocationPrice = -0.5;
    for (int R : {1, 3, (int)sol.path.size() - 1}) {
        for (int seed = 0; seed < 3; ++seed) {
            setVertexPrices(seed);
            double best = std::numeric_limits<double>::max();
            for (const segment& s : allSegments()) best = std::min(best, firstStageReducedCost(R, s, reallocationPrice));

            FirstStageMinimumPrice solver(vertexPrices, reallocationPrice, 0.0, P, c, g, sol, R);
            Reallocation r = solver.solve();
            EXPECT_EQ(r.R, R);
            ASSERT_TRUE(path_engine::is_node(r.S.source));
            EXPECT_NEAR(r.value, best, 1e-9) << "R=" << R << ", seed " << seed;
            EXPECT_NEAR(r.value, firstStageReducedCost(R, r.S, reallocationPrice), 1e-9)
                << "R=" << R << ", seed " << seed << ": value is not the reduced cost of {"
                << r.S.source << "," << r.S.target << "}";
        }
    }
}

TEST_F(minimum_price_fixture, SecondStageReturnsMinimumReducedCost) {
    const double reallocationPrice = -0.5, lengthPrice = -0.3;
    for (int R : {1, 3, (int)sol.path.size() - 1}) {
        for (int seed = 0; seed < 3; ++seed) {
            setVertexPrices(seed);
            double best = std::numeric_limits<double>::max();
            for (const segment& s : allSegments())
                best = std::min(best, secondStageReducedCost(R, s, reallocationPrice, lengthPrice));

            SecondStageMinimumPrice solver(vertexPrices, reallocationPrice, lengthPrice, P, c, g, sol, R);
            Reallocation r = solver.solve();
            EXPECT_EQ(r.R, R);
            ASSERT_TRUE(path_engine::is_node(r.S.source));
            EXPECT_NEAR(r.value, best, 1e-9) << "R=" << R << ", seed " << seed;
            EXPECT_NEAR(r.value, secondStageReducedCost(R, r.S, reallocationPrice, lengthPrice), 1e-9)
                << "R=" << R << ", seed " << seed << ": value is not the reduced cost of {"
                << r.S.source << "," << r.S.target << "}";
        }
    }
}

// solve() must not modify the solution it prices against.
TEST_F(minimum_price_fixture, SolveLeavesSolutionUnchanged) {
    setVertexPrices(0);
    const std::vector<segment> path = sol.path;
    const Eigen::VectorXd coverage = sol.coverage;
    const cost_t cost = sol.cost;
    FirstStageMinimumPrice(vertexPrices, -0.5, 0.0, P, c, g, sol, 3).solve();
    SecondStageMinimumPrice(vertexPrices, -0.5, -0.3, P, c, g, sol, 3).solve();
    EXPECT_EQ(sol.path, path);
    EXPECT_TRUE(sol.coverage.isApprox(coverage, 1e-12));
    EXPECT_EQ(sol.cost, cost);
}
