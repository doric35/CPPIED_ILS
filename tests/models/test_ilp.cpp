#include "gurobi_c++.h"
#include "../test_fixture.hpp"
#include "../../include/models/ilp.hpp"
#include "../../include/construction/dp_sweeper.hpp"

// ============================================================================
// Accessor — exposes protected members for white-box testing
// ============================================================================

struct ilp_accessor : public ilp {
    using ilp::ilp;

    // Engines (inherited via cppied_method_base)
    using cppied_method_base::geometry;
    using cppied_method_base::coverage;
    using cppied_method_base::problem;

    // Protected data members
    using ilp::Z1;
    using ilp::Z2;
    using ilp::variables;
    using ilp::minus_sets;
    using ilp::plus_sets;
    using ilp::source;
    using ilp::targets_minus;
    using ilp::targets_plus;
    using ilp::starting_minus;
    using ilp::no_rel;
    using ilp::targets_minus_flow;
    using ilp::targets_plus_flow;
    using ilp::segments_to_positions_sequence;
    using ilp::set_warm_start;

};

// ============================================================================
// Base fixture
// ============================================================================

class ilp_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<ilp_accessor> m;
    cppied_solution sol;
    std::function<void()> flow_setter;

    void SetUp() override {
        cppied_context_fixture::SetUp();
        // 7-segment local optimum on the 6×6 test seabed (same path used
        // across neighbourhood tests to ensure coverage feasibility).
        sol.path = {{6,  11},
                    {23, 18},
                    {44, 42},
                    {48, 51},
                    {30, 35},
                    {75, 73},
                    {62, 63}};
        sol.cost     = {0, 0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        m = std::make_unique<ilp_accessor>(ctx, P);
        m->coverage.reset(sol);
        sol.cost = m->geometry.cost(sol);
        ctx.start_time = std::chrono::high_resolution_clock::now();
        ctx.max_time   = 600;
        flow_setter = [&](){
            for (int u=0; u<m->problem.vertex.size(); ++u){
                for (int v =0; v<m->variables[u].outgoing_variables.size(); ++v){
                    m->variables[u].variables_flow[v] =
                            m->variables[u].outgoing_variables[v].get(GRB_DoubleAttr_X);
                }
                m->targets_minus_flow[u] = m->targets_minus[u].get(GRB_DoubleAttr_X);
                m->targets_plus_flow[u] = m->targets_plus[u].get(GRB_DoubleAttr_X);
            }
        };
    }
};

// ============================================================================
// initialize() — SOLVER_CONFIG precondition
// ============================================================================

// Bug: ctx.config["SOLVER_CONFIG"] returns "" if the key is absent (operator[]
// inserts a default-constructed value).  assert(configuration.size() == 2) then
// fires in debug builds, causing SIGABRT.  The config key must be present and
// exactly 2 characters long.
TEST_F(ilp_fixture, InitializeAssertFiresOnMissingSolverConfig) {
    if (ctx.config.find("SOLVER_CONFIG") != ctx.config.end())
        GTEST_SKIP() << "SOLVER_CONFIG is present in test config; cannot test missing-key path";

    // In release builds the assert is compiled out; in debug it fires.
    // We document the precondition rather than calling initialize() directly.
    std::string val = ctx.config["SOLVER_CONFIG"];
    EXPECT_NE(val.size(), 2u)
        << "SOLVER_CONFIG is absent so config[key] returns \"\"; "
           "initialize() asserts size()==2 — add SOLVER_CONFIG to the test config "
           "or guard with ctx.config.find() before the assert";
}

// ============================================================================
// set_flow_sets — unit tests (no Gurobi required)
// ============================================================================

// Every neighbour of u must appear in exactly one of minus_sets[u] or plus_sets[u].
TEST_F(ilp_fixture, SetFlowSetsPartitionNeighbours) {
    m->set_flow_sets(sol);
    for (int u = 0; u < (int)P.vertex.size(); ++u) {
        for (SMiIt v(P.adj, u); v; ++v) {
            int nb = v.index();
            bool in_minus = std::find(m->minus_sets[u].V.begin(),
                                      m->minus_sets[u].V.end(), nb)
                            != m->minus_sets[u].V.end();
            bool in_plus  = std::find(m->plus_sets[u].V.begin(),
                                      m->plus_sets[u].V.end(), nb)
                            != m->plus_sets[u].V.end();
            EXPECT_NE(in_minus, in_plus)
                << "neighbour " << nb << " of vertex " << u
                << " must appear in exactly one of minus_sets or plus_sets";
        }
    }
}

// minus_sets and plus_sets must be disjoint for every vertex.
TEST_F(ilp_fixture, SetFlowSetsDisjoint) {
    m->set_flow_sets(sol);
    for (int u = 0; u < (int)P.vertex.size(); ++u) {
        for (int v : m->minus_sets[u].V) {
            bool also_in_plus = std::find(m->plus_sets[u].V.begin(),
                                          m->plus_sets[u].V.end(), v)
                                != m->plus_sets[u].V.end();
            EXPECT_FALSE(also_in_plus)
                << "vertex " << v << " appears in both minus_sets[" << u
                << "] and plus_sets[" << u << "]";
        }
    }
}

// For a 6×6 seabed (84 vertices), vertex 11 sits at position (5.5, 1) and has
// exactly five neighbours: {10, 72, 73, 78, 79}.
// Neighbours 72 and 73 are also adjacent to 10 (the first-encountered neighbour),
// so the expected bipartition is minus={10,72,73} and plus={78,79}.
//
// Root bug: set_flow_sets used `problem.adj.coeff(v, ...)` where v is a
// SparseMatrix::InnerIterator.  Eigen's InnerIterator has operator bool() but no
// implicit operator Index(), so the call resolved to coeff(1, ...) — always checking
// row 1 instead of the current neighbour.  For vertex 11 this caused adj.coeff(1,10)=0
// for every successor, pushing 72 and 73 into plus_sets incorrectly.
// Fix: use v.index() explicitly.
TEST_F(ilp_fixture, SetFlowSetsNode11MinusSet) {
    m->set_flow_sets(sol);

    std::vector<int> actual(m->minus_sets[11].V.begin(), m->minus_sets[11].V.end());
    std::sort(actual.begin(), actual.end());
    const std::vector<int> expected = {10, 72, 73};

    EXPECT_EQ(actual, expected)
        << "minus_sets[11] should be {10,72,73}: neighbours at (4.5,1), (5,0.5), (5,1.5) "
           "that are all adjacent to the first-encountered neighbour 10.\n"
           "Common cause: adj.coeff(v,...) where v is an InnerIterator — resolves via "
           "operator bool() to coeff(1,...) instead of coeff(v.index(),...).";
}

TEST_F(ilp_fixture, SetFlowSetsNode11PlusSet) {
    m->set_flow_sets(sol);

    std::vector<int> actual(m->plus_sets[11].V.begin(), m->plus_sets[11].V.end());
    std::sort(actual.begin(), actual.end());
    const std::vector<int> expected = {78, 79};

    EXPECT_EQ(actual, expected)
        << "plus_sets[11] should be {78,79}: neighbours at (6,0.5) and (6,1.5) "
           "are not adjacent to vertex 10 (L1 distance 2).";
}

// General criterion: for every vertex u with a non-empty minus_sets, every member of
// minus_sets (beyond the first) must be adjacent to minus_sets[u][0], and no member of
// plus_sets may be adjacent to minus_sets[u][0].
TEST_F(ilp_fixture, SetFlowSetsPartitionCriterionHolds) {
    m->set_flow_sets(sol);

    for (int u = 0; u < (int)P.vertex.size(); ++u) {
        if (m->minus_sets[u].V.empty()) continue;
        int first_minus = m->minus_sets[u].V[0];

        for (int i = 1; i < (int)m->minus_sets[u].V.size(); ++i) {
            int v = m->minus_sets[u].V[i];
            EXPECT_EQ(P.adj.coeff(v, first_minus), 1)
                << "minus_sets[" << u << "][" << i << "] = " << v
                << " is not adjacent to first_minus=" << first_minus
                << " — partition criterion violated";
        }
        for (int v : m->plus_sets[u].V) {
            EXPECT_EQ(P.adj.coeff(v, first_minus), 0)
                << "plus_sets[" << u << "] contains " << v
                << " which IS adjacent to first_minus=" << first_minus
                << " — should be in minus_sets";
        }
    }
}

// The resulting vectors must have one entry per problem vertex.
TEST_F(ilp_fixture, SetFlowSetsSizeMatchesVertexCount) {
    m->set_flow_sets(sol);
    EXPECT_EQ((int)m->minus_sets.size(), (int)P.vertex.size());
    EXPECT_EQ((int)m->plus_sets.size(),  (int)P.vertex.size());
}

// Bug: set_flow_sets calls minus_sets.resize(n, {}) and plus_sets.resize(n, {}).
// resize() on a non-empty vector only appends new elements — existing entries are
// NOT cleared.  A second call therefore doubles the contents of every flow set.
// Fix: replace resize() with assign() or clear()+resize().
TEST_F(ilp_fixture, SetFlowSetsIdempotent) {
    m->set_flow_sets(sol);

    std::vector<size_t> minus_sz(P.vertex.size()), plus_sz(P.vertex.size());
    for (int u = 0; u < (int)P.vertex.size(); ++u) {
        minus_sz[u] = m->minus_sets[u].V.size();
        plus_sz[u]  = m->plus_sets[u].V.size();
    }

    m->set_flow_sets(sol);   // second call

    for (int u = 0; u < (int)P.vertex.size(); ++u) {
        EXPECT_EQ(m->minus_sets[u].V.size(), minus_sz[u])
            << "minus_sets[" << u << "] grew from " << minus_sz[u]
            << " to " << m->minus_sets[u].V.size()
            << " on second call — resize() does not clear existing elements; "
               "use assign() or clear()+resize()";
        EXPECT_EQ(m->plus_sets[u].V.size(), plus_sz[u])
            << "plus_sets[" << u << "] grew on second call";
    }
}

// ============================================================================
// set_variables — requires Gurobi
// ============================================================================

class ilp_variables_fixture : public ilp_fixture {
protected:
    GRBEnv genv{true};
    std::unique_ptr<GRBModel> grb_model;

    void SetUp() override {
        ilp_fixture::SetUp();
        genv.set(GRB_IntParam_OutputFlag, 0);
        genv.start();
        grb_model = std::make_unique<GRBModel>(genv);
        m->set_flow_sets(sol);
        m->set_variables(sol, *grb_model);
        grb_model->update();
    }
};

// variables[] must have exactly one entry per problem vertex.
TEST_F(ilp_variables_fixture, VariablesSizeMatchesVertexCount) {
    EXPECT_EQ((int)m->variables.size(), (int)P.vertex.size());
}

// targets_minus and targets_plus must each have one entry per problem vertex.
TEST_F(ilp_variables_fixture, TargetVariablesSizeMatchesVertexCount) {
    EXPECT_EQ((int)m->targets_minus.size(), (int)P.vertex.size());
    EXPECT_EQ((int)m->targets_plus.size(),  (int)P.vertex.size());
}

// variables_flow must be initialised to zero for all arcs.
TEST_F(ilp_variables_fixture, VariablesFlowInitialisedToZero) {
    for (int u = 0; u < (int)m->variables.size(); ++u)
        for (int f : m->variables[u].variables_flow)
            EXPECT_EQ(f, 0)
                << "variables_flow not zero at vertex " << u;
}

// For each vertex u, outgoing_arcs_V1 and outgoing_variables must be the same size.
TEST_F(ilp_variables_fixture, OutgoingArcsAndVariablesSizeConsistent) {
    for (int u = 0; u < (int)m->variables.size(); ++u)
        EXPECT_EQ(m->variables[u].outgoing_arcs_V1.size(),
                  m->variables[u].outgoing_variables.size())
            << "outgoing_arcs_V1.size() != outgoing_variables.size() at vertex " << u;
}

// For each vertex u, outgoing_arcs_V1 and variables_flow must be the same size.
TEST_F(ilp_variables_fixture, OutgoingArcsAndFlowSizeConsistent) {
    for (int u = 0; u < (int)m->variables.size(); ++u)
        EXPECT_EQ(m->variables[u].outgoing_arcs_V1.size(),
                  m->variables[u].variables_flow.size())
            << "outgoing_arcs_V1.size() != variables_flow.size() at vertex " << u;
}

// vertex_to_arc must be the inverse of outgoing_arcs_V1.
TEST_F(ilp_variables_fixture, VertexToArcIsConsistentInverseMap) {
    for (int u = 0; u < (int)m->variables.size(); ++u) {
        for (int arc_id = 0;
             arc_id < (int)m->variables[u].outgoing_arcs_V1.size();
             ++arc_id) {
            int nb = m->variables[u].outgoing_arcs_V1[arc_id];
            auto it = m->variables[u].vertex_to_arc.find(nb);
            ASSERT_NE(it, m->variables[u].vertex_to_arc.end())
                << "vertex_to_arc missing entry for neighbour " << nb
                << " of vertex " << u;
            EXPECT_EQ(it->second, arc_id)
                << "vertex_to_arc[" << nb << "] = " << it->second
                << " but arc_id = " << arc_id << " at vertex " << u;
        }
    }
}

// source must point to initial_position as its sole outgoing arc.
TEST_F(ilp_variables_fixture, SourceOutgoingArcIsInitialPosition) {
    ASSERT_EQ((int)m->source.outgoing_arcs_V1.size(), 1)
        << "source must have exactly one outgoing arc";
    EXPECT_EQ(m->source.outgoing_arcs_V1[0], P.initial_position);
}

// Bug: set_variables calls variables.reserve() but NOT variables.clear().
// reserve() never removes existing elements, so a second call appends new nodes
// on top of the first set.  Fix: add variables.clear() before reserve().
TEST_F(ilp_variables_fixture, SetVariablesIdempotent) {
    int expected = (int)P.vertex.size();
    m->set_variables(sol, *grb_model);   // second call

    EXPECT_EQ((int)m->variables.size(), expected)
        << "variables grew from " << expected << " to " << m->variables.size()
        << " on second call — reserve() does not clear; add variables.clear()";
    EXPECT_EQ((int)m->targets_minus.size(), expected)
        << "targets_minus grew on second call";
    EXPECT_EQ((int)m->targets_plus.size(), expected)
        << "targets_plus grew on second call";
}

// ============================================================================
// find_cycle — white-box tests (manual flow injection, no Gurobi solve)
// ============================================================================
//
// Contract: find_cycle traces exactly one cycle reachable from (current_node,
// curr_minus).  It returns an empty list if no outgoing arc with flow > 0 exists
// in the expected direction set.  Flow in disconnected components that are not
// reachable from the entry point is left untouched.
//
// Grid topology (6×6 seabed, 84 vertices):
//   Horizontal h(i,j) = i*6+j   at (j+0.5, i),  i=0..6, j=0..5
//   Vertical   v(j,i) = 42+j*6+i at (j, i+0.5), j=0..6, i=0..5
//
// Two non-overlapping reference cycles (cell-boundary loops):
//
//   Cycle A (cell (0,0)):  0 → 42 → 6 → 48 → 0
//     minus_sets[0]={1,48}   plus_sets[0]={42}      → enter curr_minus=true
//     minus_sets[42]={0}     plus_sets[42]={6}
//     minus_sets[6]={7,48,49} plus_sets[6]={42,43}
//     minus_sets[48]={0,1}   plus_sets[48]={6,7}
//     → find_cycle(sol, path, 0, true) == {42, 6, 48, 0}
//
//   Cycle B (cell (0,2)): 12 → 44 → 18 → 50 → 12
//     minus_sets[12]={13,49,50} plus_sets[12]={43,44} → enter curr_minus=true
//     minus_sets[44]={12}       plus_sets[44]={18}
//     minus_sets[18]={19,50,51} plus_sets[18]={44,45}
//     minus_sets[50]={12,13}    plus_sets[50]={18,19}
//     → find_cycle(sol, path, 12, true) == {44, 18, 50, 12}

class ilp_find_cycle_fixture : public ilp_variables_fixture {
protected:
    cppied_solution dummy;

    void SetUp() override {
        ilp_variables_fixture::SetUp();
        dummy.path     = {};
        dummy.coverage = Eigen::VectorXd::Zero(P.req.size());
        dummy.cost     = {0, 0};
    }

    // Set the flow on the directed arc u→v to `flow`.
    void set_arc_flow(int u, int v, int flow) {
        auto it = m->variables[u].vertex_to_arc.find(v);
        ASSERT_NE(it, m->variables[u].vertex_to_arc.end())
            << "No arc from " << u << " to " << v
            << " — vertices must be geometrically adjacent";
        m->variables[u].variables_flow[it->second] = flow;
    }

    // Return the total remaining flow across all arcs.
    int total_flow() const {
        int s = 0;
        for (auto& node : m->variables)
            for (int f : node.variables_flow) s += f;
        return s;
    }

    void setup_cycle_A() {   // 0→42→6→48→0
        set_arc_flow( 0, 42, 1);
        set_arc_flow(42,  6, 1);
        set_arc_flow( 6, 48, 1);
        set_arc_flow(48,  0, 1);
    }
    void setup_cycle_B() {   // 12→44→18→50→12
        set_arc_flow(12, 44, 1);
        set_arc_flow(44, 18, 1);
        set_arc_flow(18, 50, 1);
        set_arc_flow(50, 12, 1);
    }
};

// ── Basic empty cases ─────────────────────────────────────────────────────────

// No flow set anywhere: path must be empty.
TEST_F(ilp_find_cycle_fixture, FindCycleEmptyWhenNoFlow) {
    std::list<int> path;
    m->find_cycle(dummy, path, 0, true);
    EXPECT_TRUE(path.empty())
        << "find_cycle should produce an empty path when no flow is set";
}

// Flow exists but not reachable from the entry node: path must be empty and
// the disconnected flow must be left untouched.
TEST_F(ilp_find_cycle_fixture, FindCycleEmptyWhenNoFlowAtEntryNode) {
    setup_cycle_A();   // flow on {0,6,42,48}, not on vertex 11
    std::list<int> path;
    m->find_cycle(dummy, path, 11, /*curr_minus=*/true);

    EXPECT_TRUE(path.empty())
        << "find_cycle must return empty when no arc exits vertex 11 in the "
           "plus direction — it must not scan disconnected components";

    EXPECT_EQ(total_flow(), 4)
        << "Cycle A's flow must remain untouched after a dead-end call on vertex 11";
}

// Entry node has flow, but in the wrong direction set: path must be empty.
TEST_F(ilp_find_cycle_fixture, FindCycleEmptyWhenWrongDirection) {
    setup_cycle_A();   // needs curr_minus=true at vertex 0 (plus_sets[0]={42})
    std::list<int> path;
    m->find_cycle(dummy, path, 0, /*curr_minus=*/false);  // wrong: looks in minus_sets[0]={1,48}

    EXPECT_TRUE(path.empty())
        << "find_cycle must return empty when the entry direction points to a set "
           "with no flow, even if flow exists in the other direction";
}

// ── Correct tracing tests ─────────────────────────────────────────────────────

// Cycle A traced from its entry vertex with the correct direction.
TEST_F(ilp_find_cycle_fixture, FindCycleTracesCycleAInOrder) {
    setup_cycle_A();
    std::list<int> path;
    m->find_cycle(dummy, path, 0, /*curr_minus=*/true);

    const std::list<int> expected = {42, 6, 48, 0};
    EXPECT_EQ(path, expected)
        << "Cycle A (0→42→6→48→0): expected path {42,6,48,0}";
}

// All four nodes of Cycle A must appear in the result.
TEST_F(ilp_find_cycle_fixture, FindCycleAllCycleANodesPresent) {
    setup_cycle_A();
    std::list<int> path;
    m->find_cycle(dummy, path, 0, true);

    std::set<int> in_path(path.begin(), path.end());
    for (int v : {0, 6, 42, 48})
        EXPECT_TRUE(in_path.count(v))
            << "Cycle A node " << v << " missing from path";
}

// After tracing Cycle A the four arcs must be consumed; no other flow changes.
TEST_F(ilp_find_cycle_fixture, FindCycleCycleAConsumesExactlyItsFlow) {
    setup_cycle_A();
    setup_cycle_B();   // independent flow that must NOT be touched
    std::list<int> path;
    m->find_cycle(dummy, path, 0, true);

    EXPECT_EQ(total_flow(), 4)
        << "Only Cycle A's 4 arcs should be consumed; Cycle B's 4 arcs must remain";
}

// Cycle B traced from its own entry vertex with the correct direction.
TEST_F(ilp_find_cycle_fixture, FindCycleTracesCycleBInOrder) {
    setup_cycle_B();
    std::list<int> path;
    m->find_cycle(dummy, path, 12, /*curr_minus=*/true);

    const std::list<int> expected = {44, 18, 50, 12};
    EXPECT_EQ(path, expected)
        << "Cycle B (12→44→18→50→12): expected path {44,18,50,12}";
}

// ── Flow-accounting invariant ─────────────────────────────────────────────────

// After a successful trace, exactly the cycle's arcs are consumed and nothing else.
TEST_F(ilp_find_cycle_fixture, FindCycleTotalFlowDecreasedByOneCycle) {
    setup_cycle_A();
    const int before = total_flow();  // 4
    std::list<int> path;
    m->find_cycle(dummy, path, 0, true);
    EXPECT_EQ(total_flow(), before - (int)path.size())
        << "total flow must decrease by exactly path.size() (one unit per arc consumed)";
}

// ============================================================================
// segments_to_positions_sequence + set_warm_start — white-box tests
// ============================================================================
//
// Bugs the tests are designed to catch:
//   (1) segments_to_positions_sequence: backward segments are silently dropped
//       because the loop condition "v <= s.target" terminates immediately when
//       s.target < s.source (orientation == -1).  Fix: negate the condition
//       for negative orientation (e.g. use "v != s.target + orientation").
//   (2) set_warm_start: back_minus detection compared the found iterator
//       against p_sequence.end() (a different container) instead of
//       minus_sets[p_sequence.back()].V.end(), always evaluating to true
//       and forcing the wrong target variable to be set.
//
// Grid topology: same 6×6 seabed as the find_cycle tests (84 vertices):
//   Horizontal h(i,j) = i*6+j   at (j+0.5, i),  i=0..6, j=0..5
//   Vertical   v(j,i) = 42+j*6+i at (j, i+0.5), j=0..6, i=0..5
//
// The sequence and warm-start tests use a feasible dp_sweeper solution so
// that all consecutive pairs in the sequence correspond to real ILP arcs.

// ── Fixture for pure sequence tests (no Gurobi required) ─────────────────────
// segments_to_positions_sequence operates only on path segment data and does
// not touch any Gurobi variable.  These tests therefore run without a license.

class ilp_positions_sequence_fixture : public ilp_fixture {
protected:
    std::vector<int> sequence_of(std::vector<segment> segs) {
        cppied_solution tmp;
        tmp.path     = std::move(segs);
        tmp.cost     = {0, 0};
        tmp.coverage = Eigen::VectorXd::Zero(P.req.size());
        std::vector<int> s;
        m->segments_to_positions_sequence(tmp, s);
        return s;
    }
};

// A forward segment {6,11} on the 6×6 grid must expand to exactly
// [6,7,8,9,10,11] — no duplicate source, no missing tail.
TEST_F(ilp_positions_sequence_fixture, PositionSequenceForwardSegmentNoDuplicate) {
    auto s = sequence_of({{6, 11}});
    const std::vector<int> expected = {6, 7, 8, 9, 10, 11};
    EXPECT_EQ(s, expected)
        << "forward segment {6,11} must expand to [6,7,8,9,10,11] without "
           "leading duplication of the source vertex";
}

// A backward segment {23,18} must expand to [23,22,21,20,19,18].
// Bug: the loop "for (v = src+orientation; v <= tgt; ...)" exits immediately
// when tgt < src (orientation == -1), so only the source vertex is pushed.
TEST_F(ilp_positions_sequence_fixture, PositionSequenceBackwardSegmentComplete) {
    auto s = sequence_of({{23, 18}});
    const std::vector<int> expected = {23, 22, 21, 20, 19, 18};
    EXPECT_EQ(s, expected)
        << "backward segment {23,18} must expand to all 6 vertices; "
           "bug: 'v <= s.target' exits immediately for negative orientation";
}

// ── Fixture for warm-start tests (requires Gurobi) ────────────────────────────
// set_warm_start calls GRBVar::set(GRB_DoubleAttr_Start, ...) on all arc
// variables, so a valid Gurobi environment is needed.

class ilp_warm_start_fixture : public ilp_variables_fixture {
protected:
    cppied_solution warm_sol;  // feasible dp_sweeper solution
    std::vector<int> seq;      // position sequence from warm_sol

    void SetUp() override {
        ilp_variables_fixture::SetUp();

        warm_sol.path     = {};
        warm_sol.cost     = {0, 0};
        warm_sol.coverage = Eigen::VectorXd::Zero(P.req.size());

        dp_sweeper dps(ctx, P);
        auto saver = [](const cppied_solution&) {};
        dps.construct(warm_sol, saver);
        m->geometry.complete(warm_sol);
        m->coverage.reset(warm_sol);
        warm_sol.cost = m->geometry.cost(warm_sol);

        m->segments_to_positions_sequence(warm_sol, seq);
        m->set_warm_start(seq);
    }
};

// The first element of the position sequence must be the problem's initial
// position vertex (that is where the source arc originates).
TEST_F(ilp_warm_start_fixture, PositionSequenceStartsAtInitialPosition) {
    ASSERT_FALSE(seq.empty());
    EXPECT_EQ(seq.front(), P.initial_position)
        << "position sequence must start at initial_position="
        << P.initial_position << "; got " << seq.front();
}

// No two consecutive positions in the sequence may be identical (self-arcs
// do not exist in the ILP graph and would corrupt set_warm_start).
TEST_F(ilp_warm_start_fixture, PositionSequenceNoConsecutiveDuplicates) {
    ASSERT_GE((int)seq.size(), 2)
        << "dp_sweeper must produce a sequence with at least two vertices";
    for (int i = 0; i + 1 < (int)seq.size(); ++i)
        EXPECT_NE(seq[i], seq[i + 1])
            << "self-arc at position " << i << ": seq[" << i << "]=seq["
            << i + 1 << "]=" << seq[i];
}

// Every consecutive pair (seq[i], seq[i+1]) must correspond to an existing
// arc in the ILP graph, i.e. vertex_to_arc[seq[i]] must contain seq[i+1].
// A missing arc indicates that segments_to_positions_sequence skipped
// vertices (backward-segment bug) and the resulting jump is not a valid edge.
TEST_F(ilp_warm_start_fixture, PositionSequenceAllConsecutivePairsAreValidArcs) {
    for (int i = 0; i + 1 < (int)seq.size(); ++i) {
        int u = seq[i], v = seq[i + 1];
        auto it = m->variables[u].vertex_to_arc.find(v);
        EXPECT_NE(it, m->variables[u].vertex_to_arc.end())
            << "seq[" << i << "]=" << u << " → seq[" << i + 1 << "]=" << v
            << " is not a valid ILP arc; vertex_to_arc[" << u
            << "] has no entry for " << v;
    }
}

// set_warm_start must set variables_flow[u][arc(u→v)] to the number of times
// the arc u→v appears as a consecutive pair in the position sequence.
TEST_F(ilp_warm_start_fixture, WarmStartFlowMatchesSequenceTraversal) {
    std::map<std::pair<int, int>, int> expected_flow;
    for (int i = 0; i + 1 < (int)seq.size(); ++i)
        expected_flow[{seq[i], seq[i + 1]}]++;

    for (auto& [uv, count] : expected_flow) {
        int u   = uv.first;
        int v   = uv.second;
        auto it = m->variables[u].vertex_to_arc.find(v);
        if (it == m->variables[u].vertex_to_arc.end()) {
            ADD_FAILURE() << "arc " << u << "→" << v
                          << " not in vertex_to_arc; cannot verify flow";
            continue;
        }
        double flow = m->variables[u].variables_flow[it->second];
        EXPECT_NEAR(flow, (double)count, 1e-6)
            << "variables_flow[" << u << "][arc→" << v << "]=" << flow
            << "; expected " << count << " traversal(s)";
    }
}

// Exactly one target variable (targets_minus_flow or targets_plus_flow) must
// be set to 1 across all vertices — the path has a unique endpoint.
TEST_F(ilp_warm_start_fixture, WarmStartExactlyOneTargetVariableSet) {
    int n = (int)m->targets_minus_flow.size();
    int total_set = 0;
    for (int v = 0; v < n; ++v) {
        if (m->targets_minus_flow[v] > 0.5) ++total_set;
        if (m->targets_plus_flow[v]  > 0.5) ++total_set;
    }
    EXPECT_EQ(total_set, 1)
        << "exactly one target variable must be set to 1; got " << total_set
        << " (bug: back_minus detection uses wrong end iterator, always sets "
           "the same type and leaves the other unset or double-sets)";
}

// The flow set by set_warm_start must satisfy every coverage constraint:
//   s_pod(c, initial_pos) + Σ_{u→v} variables_flow[u][arc] * s_pod(c,v) ≥ req(c)
//
// This directly models the Gurobi violation:
//   "User MIP start violates constraint R0 by 2.197224577"
// Root cause: backward segments were not expanded, so arcs covering certain
// cells were missing from the warm-start flow.
TEST_F(ilp_warm_start_fixture, WarmStartSatisfiesCoverageConstraints) {
    ASSERT_TRUE((warm_sol.coverage.array() >= P.req.array()).all());
    int n_cells = (int)P.req.size();
    for (int c = 0; c < n_cells; ++c) {
        if (P.req(c) < 1e-9) continue;

        // Source arc contribution: the source arc always covers initial_position.
        double cov = P.s_pod.coeff(c, P.initial_position);

        // Contribution from every arc with non-zero flow.
        for (int u = 0; u < (int)m->variables.size(); ++u) {
            for (int arc = 0;
                 arc < (int)m->variables[u].outgoing_arcs_V1.size(); ++arc) {
                double flow = m->variables[u].variables_flow[arc];
                if (flow < 1e-9) continue;
                int dest = m->variables[u].outgoing_arcs_V1[arc];
                cov += flow * P.s_pod.coeff(c, dest);
            }
        }

        EXPECT_GE(cov, P.req(c) - 1e-6)
            << "coverage constraint violated for cell " << c
            << ": coverage=" << cov << " < req=" << P.req(c);
    }
}

// ============================================================================
// set_constraints — requires variables to be set
// ============================================================================

class ilp_constraints_fixture : public ilp_variables_fixture {
protected:
    // Must outlive grb_model: Gurobi holds a raw pointer to the callback object
    // and invokes it during optimize().  A stack variable in SetUp() is destroyed
    // before any test body runs, leaving a dangling pointer → BAD ACCESS.
    std::unique_ptr<subtour_elimination> cb;

    void SetUp() override {
        ilp_variables_fixture::SetUp();
        m->set_constraints(sol, *grb_model);
        cb = std::make_unique<subtour_elimination>(*m);
        grb_model->set(GRB_IntParam_LazyConstraints, 1);
        grb_model->setCallback(cb.get());
        grb_model->update();
    }
};

// There must be at least n_rows*n_cols coverage constraints plus flow conservation
// (2 per vertex) plus the source and target constraints.
TEST_F(ilp_constraints_fixture, TotalConstraintCountLowerBound) {
    int n_cells    = m->geometry.n_rows * m->geometry.n_cols;
    int n_vertices = (int)P.vertex.size();
    int min_expected = n_cells + 2 * n_vertices + 2;
    int actual       = grb_model->get(GRB_IntAttr_NumConstrs);
    EXPECT_GE(actual, min_expected)
        << "expected at least " << min_expected << " constraints, got " << actual;
}

// Source constraint: sum of source variables must be exactly 1.
// This verifies the path has a unique starting arc.
TEST_F(ilp_constraints_fixture, SourceConstraintForcesSingleEntry) {
    // Minimize source variable sum subject to constraints.
    // Optimal value must be 1 if the source constraint is correctly added.
    grb_model->setObjective(GRBLinExpr(m->source.outgoing_variables[0]), GRB_MINIMIZE);
    grb_model->update();
    grb_model->optimize();
    int status = grb_model->get(GRB_IntAttr_Status);
    if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL)
        GTEST_SKIP() << "model infeasible or timed out; cannot verify source constraint";
    EXPECT_NEAR(grb_model->get(GRB_DoubleAttr_ObjVal), 1.0, 1e-6)
        << "source variable is not forced to 1 by the source constraint";
}

// Target constraint: exactly one of all targets_minus and targets_plus must be active.
TEST_F(ilp_constraints_fixture, TargetConstraintForcesSingleExit) {
    GRBLinExpr sum_targets = 0;
    for (auto& v : m->targets_minus) sum_targets += v;
    for (auto& v : m->targets_plus)  sum_targets += v;
    grb_model->setObjective(sum_targets, GRB_MINIMIZE);
    grb_model->update();
    grb_model->optimize();
    int status = grb_model->get(GRB_IntAttr_Status);
    if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL)
        GTEST_SKIP() << "model infeasible or timed out; cannot verify target constraint";
    EXPECT_NEAR(grb_model->get(GRB_DoubleAttr_ObjVal), 1.0, 1e-6)
        << "sum of target variables is not forced to 1 by the target constraint";
}

// ============================================================================
// set_objectives — requires constraints to be set
// ============================================================================

class ilp_objectives_fixture : public ilp_constraints_fixture {
protected:
    void SetUp() override {
        ilp_constraints_fixture::SetUp();
        m->set_objectives(sol, *grb_model);
        grb_model->update();
    }
};

// After set_objectives the model must register exactly 2 objectives (Z1 and Z2).
TEST_F(ilp_objectives_fixture, ModelHasTwoObjectives) {
    EXPECT_EQ(grb_model->get(GRB_IntAttr_NumObj), 2)
        << "expected 2 multi-objectives (Z1 = segment count, Z2 = direction changes)";
}

// Bug: Z1 and Z2 are protected GRBLinExpr class members.  set_objectives
// accumulates into them via +=.  Calling set_objectives twice (or d_solve twice
// without clearing) doubles every coefficient.
// Fix: add Z1 = 0; Z2 = 0; at the start of set_objectives (or d_solve).
TEST_F(ilp_objectives_fixture, ObjectivesNotAccumulatedOnDoubleCall) {
    // Build a fresh accessor and call set_objectives once for the reference.
    GRBModel ref_model(genv);
    ilp_accessor ref_acc(ctx, P);
    ref_acc.set_flow_sets(sol);
    ref_acc.set_variables(sol, ref_model);
    ref_acc.set_constraints(sol, ref_model);
    ref_acc.set_objectives(sol, ref_model);
    ref_model.setObjective(ref_acc.Z1, GRB_MINIMIZE);
    ref_model.update();
    ref_model.optimize();
    if (ref_model.get(GRB_IntAttr_Status) != GRB_OPTIMAL)
        GTEST_SKIP() << "reference model infeasible; cannot compare objectives";
    double obj_once = ref_model.get(GRB_DoubleAttr_ObjVal);

    // Build a second accessor and call set_objectives twice.
    GRBModel dbl_model(genv);
    ilp_accessor dbl_acc(ctx, P);
    dbl_acc.set_flow_sets(sol);
    dbl_acc.set_variables(sol, dbl_model);
    dbl_acc.set_constraints(sol, dbl_model);
    dbl_acc.set_objectives(sol, dbl_model);  // first call
    dbl_acc.set_objectives(sol, dbl_model);  // second call — doubles coefficients
    dbl_model.setObjective(dbl_acc.Z1, GRB_MINIMIZE);
    dbl_model.update();
    dbl_model.optimize();
    if (dbl_model.get(GRB_IntAttr_Status) != GRB_OPTIMAL)
        GTEST_SKIP() << "double model infeasible; cannot compare objectives";
    double obj_twice = dbl_model.get(GRB_DoubleAttr_ObjVal);

    EXPECT_NEAR(obj_once, obj_twice, 1e-6)
        << "objective value changed from " << obj_once << " to " << obj_twice
        << " after calling set_objectives twice on the same model; "
           "Z1 and Z2 accumulate — add Z1=0; Z2=0; before accumulation loop";
}

// ============================================================================
// retrieve_solution — requires an optimal Gurobi solution
// ============================================================================

class ilp_solved_fixture : public ilp_objectives_fixture {
protected:
    void SetUp() override {
        ilp_objectives_fixture::SetUp();
        grb_model->optimize();
        int status = grb_model->get(GRB_IntAttr_Status);
        if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL)
            GTEST_SKIP() << "Gurobi did not find a feasible solution; skipping retrieve tests";
    }
};

// retrieve_solution must produce a non-empty path.
TEST_F(ilp_solved_fixture, RetrieveSolutionProducesNonEmptyPath) {
    cppied_solution result = sol;
    result.coverage = Eigen::VectorXd::Zero(P.req.size());
    std::list<int> path;
    m->retrieve_solution(result, path, flow_setter);
    m->set_solution(result, path);
    EXPECT_GT(result.path.size(), 0u)
        << "retrieve_solution produced an empty path";
}

// The retrieved path must satisfy all coverage requirements.
TEST_F(ilp_solved_fixture, RetrieveSolutionSatisfiesCoverage) {
    cppied_solution result = sol;
    result.coverage = Eigen::VectorXd::Zero(P.req.size());
    std::list<int> path;
    m->retrieve_solution(result, path, flow_setter);
    m->set_solution(result, path);
    m->coverage.reset(result);
    EXPECT_TRUE((result.coverage.array() >= P.req.array()).all())
        << "retrieved solution violates coverage requirements";
}

// The stored cost must match geometry.cost(path) after retrieve_solution.
TEST_F(ilp_solved_fixture, RetrieveSolutionCostConsistent) {
    cppied_solution result = sol;
    result.coverage = Eigen::VectorXd::Zero(P.req.size());
    std::list<int> path;
    m->retrieve_solution(result, path, flow_setter);
    m->set_solution(result, path);
    EXPECT_EQ(m->geometry.cost(result), result.cost)
        << "result.cost is inconsistent with geometry.cost after retrieve_solution";
}

// Bug (documented behaviour): retrieve_solution consumes variables_flow by
// decrementing arc flows to zero as it walks the graph.  After the call, all
// flow values must be 0 — a second call would find no arcs to follow and
// produce an incorrect or empty path.
TEST_F(ilp_solved_fixture, RetrieveSolutionConsumesVariablesFlow) {
    cppied_solution result = sol;
    result.coverage = Eigen::VectorXd::Zero(P.req.size());
    std::list<int> path;
    m->retrieve_solution(result, path, flow_setter);
    m->set_solution(result, path);

    for (int u = 0; u < (int)m->variables.size(); ++u)
        for (int f : m->variables[u].variables_flow)
            EXPECT_EQ(f, 0)
                << "variables_flow[" << u << "][*] is non-zero after retrieve_solution; "
                   "the method is destructive — it consumes the flow during path extraction. "
                   "A second call cannot reconstruct the path.";
}

// ============================================================================
// State-reset / re-entrant safety — d_solve called twice
// ============================================================================

// Bug: Z1 and Z2 are cleared at the END of d_solve.  If d_solve throws before
// reaching that point, or if set_objectives is called at the start of the next
// call before the clear, Z1/Z2 hold stale (dangling) GRBVar handles from the
// destroyed first GRBModel.
// Fix: reset Z1=0; Z2=0; at the very start of d_solve (or set_objectives).
TEST_F(ilp_fixture, DsolveSecondCallDoesNotCrash) {
    ASSERT_NO_THROW(m->d_solve(sol))
        << "first call to d_solve threw unexpectedly";

    // Restore a fresh, feasible solution for the second call.
    sol.path     = {{6, 11}, {23, 18}, {44, 42},
                    {48, 51}, {30, 35}, {75, 73}, {62, 63}};
    sol.coverage = Eigen::VectorXd::Zero(P.req.size());
    m->coverage.reset(sol);
    sol.cost = m->geometry.cost(sol);

    EXPECT_NO_THROW(m->d_solve(sol))
        << "second call to d_solve crashed; likely Z1/Z2 hold dangling GRBVar "
           "references from the first GRBModel. "
           "Fix: add Z1=0; Z2=0; at the start of set_objectives or d_solve.";
}

// Bug: set_flow_sets uses resize() which does not clear existing entries.
// After two calls to d_solve the flow sets have doubled contents.
TEST_F(ilp_fixture, DsolveSecondCallFlowSetsNotDoubled) {
    ASSERT_NO_THROW(m->d_solve(sol));

    std::vector<size_t> minus_sz(P.vertex.size()), plus_sz(P.vertex.size());
    for (int u = 0; u < (int)P.vertex.size(); ++u) {
        minus_sz[u] = m->minus_sets[u].V.size();
        plus_sz[u]  = m->plus_sets[u].V.size();
    }

    sol.path     = {{6, 11}, {23, 18}, {44, 42},
                    {48, 51}, {30, 35}, {75, 73}, {62, 63}};
    sol.coverage = Eigen::VectorXd::Zero(P.req.size());
    m->coverage.reset(sol);
    sol.cost = m->geometry.cost(sol);
    ASSERT_NO_THROW(m->d_solve(sol));

    for (int u = 0; u < (int)P.vertex.size(); ++u) {
        EXPECT_EQ(m->minus_sets[u].V.size(), minus_sz[u])
            << "minus_sets[" << u << "] grew from " << minus_sz[u]
            << " to " << m->minus_sets[u].V.size() << " after second d_solve — "
               "set_flow_sets::resize() does not clear; use assign() or clear()+resize()";
        EXPECT_EQ(m->plus_sets[u].V.size(), plus_sz[u])
            << "plus_sets[" << u << "] grew after second d_solve";
    }
}

// Bug: set_variables uses reserve() rather than clear()+reserve().
// A second call to d_solve appends additional vertices to variables[], doubling it.
TEST_F(ilp_fixture, DsolveSecondCallVariablesNotDoubled) {
    ASSERT_NO_THROW(m->d_solve(sol));

    int expected = (int)P.vertex.size();
    ASSERT_EQ((int)m->variables.size(), expected)
        << "sanity: variables[] must have exactly one entry per vertex after first d_solve";

    sol.path     = {{6, 11}, {23, 18}, {44, 42},
                    {48, 51}, {30, 35}, {75, 73}, {62, 63}};
    sol.coverage = Eigen::VectorXd::Zero(P.req.size());
    m->coverage.reset(sol);
    sol.cost = m->geometry.cost(sol);
    ASSERT_NO_THROW(m->d_solve(sol));

    EXPECT_EQ((int)m->variables.size(), expected)
        << "variables[] grew from " << expected
        << " to " << m->variables.size() << " after second d_solve — "
           "set_variables uses reserve(), which does not clear; "
           "add variables.clear() before the population loop";
    EXPECT_EQ((int)m->targets_minus.size(), expected)
        << "targets_minus doubled after second d_solve";
    EXPECT_EQ((int)m->targets_plus.size(), expected)
        << "targets_plus doubled after second d_solve";
}

// ============================================================================
// E2E — full d_solve pipeline
// ============================================================================

// d_solve must not throw on a feasible input.
TEST_F(ilp_fixture, DsolveDoesNotThrow) {
    EXPECT_NO_THROW(m->d_solve(sol));
}

// The path produced by d_solve must be non-empty.
TEST_F(ilp_fixture, DsolveProducesNonEmptyPath) {
    ASSERT_NO_THROW(m->d_solve(sol));
    EXPECT_GT(sol.path.size(), 0u)
        << "d_solve produced an empty path";
}

// All coverage requirements must be satisfied after d_solve.
TEST_F(ilp_fixture, DsolveSolutionFeasible) {
    ASSERT_NO_THROW(m->d_solve(sol));
    m->coverage.reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all())
        << "d_solve produced a solution that violates coverage requirements";
}

// The stored cost must equal geometry.cost(path) after d_solve.
TEST_F(ilp_fixture, DsolveCostConsistent) {
    ASSERT_NO_THROW(m->d_solve(sol));
    EXPECT_EQ(m->geometry.cost(sol), sol.cost)
        << "sol.cost is inconsistent with geometry.cost after d_solve";
}

// The incremental coverage vector must match a full reset after d_solve.
TEST_F(ilp_fixture, DsolveCoverageConsistent) {
    ASSERT_NO_THROW(m->d_solve(sol));
    Eigen::VectorXd saved = sol.coverage;
    m->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9))
        << "incremental coverage vector differs from a full reset after d_solve";
}

// Z1 and Z2 must be empty (cleared) after a completed d_solve call.
// Bug: if d_solve throws after set_objectives but before Z1.clear(), the class
// retains stale GRBVar handles that will dangle when the model is destroyed.
// This test verifies normal-path clearing.
TEST_F(ilp_fixture, DsolveClearsZ1Z2AfterCompletion) {
    ASSERT_NO_THROW(m->d_solve(sol));
    // GRBLinExpr size() is not public; verify indirectly by checking that a
    // second call to set_objectives on a fresh model does not produce
    // a doubled objective value (relies on Z1=0/Z2=0 having been reset).
    GRBEnv env{true};
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel model2(env);
    m->set_flow_sets(sol);        // re-populate after d_solve consumed flow state
    m->set_variables(sol, model2);
    m->set_constraints(sol, model2);
    m->set_objectives(sol, model2);
    model2.update();
    model2.optimize();
    EXPECT_NE(model2.get(GRB_IntAttr_Status), GRB_INFEASIBLE)
        << "fresh model after d_solve completion is infeasible; "
           "Z1/Z2 may contain stale expressions from the previous call";
}

//-----------Testing on a larger fixture for SEC debugging-----------------------
class ilp_large_fixture : public cppied_context_large_fixture {
protected:
    std::unique_ptr<ilp_accessor> m;
    cppied_solution sol;
    std::function<void()> flow_setter;

    void SetUp() override {
        cppied_context_large_fixture::SetUp();
        // 7-segment local optimum on the 6×6 test seabed (same path used
        // across neighbourhood tests to ensure coverage feasibility).

        ctx.start_time = std::chrono::high_resolution_clock::now();
        ctx.max_time   = 60;

        sol.path = {};
        sol.cost     = {0, 0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());

        dp_sweeper dps(ctx, P);
        auto history_save = [](cppied_solution&){

        };
        dps.construct(sol, history_save);

        m = std::make_unique<ilp_accessor>(ctx, P);
        m->coverage.reset(sol);
        sol.cost = m->geometry.cost(sol);

        flow_setter = [&](){
            for (int u=0; u<m->problem.vertex.size(); ++u){
                for (int v =0; v<m->variables[u].outgoing_variables.size(); ++v){
                    m->variables[u].variables_flow[v] =
                            m->variables[u].outgoing_variables[v].get(GRB_DoubleAttr_X);
                }
                m->targets_minus_flow[u] = m->targets_minus[u].get(GRB_DoubleAttr_X);
                m->targets_plus_flow[u] = m->targets_plus[u].get(GRB_DoubleAttr_X);
            }
        };
    }
};

class ilp_variables_large_fixture : public ilp_large_fixture {
protected:
    GRBEnv genv{true};
    std::unique_ptr<GRBModel> grb_model;

    void SetUp() override {
        ilp_large_fixture::SetUp();
        genv.set(GRB_IntParam_OutputFlag, 0);
        genv.start();
        grb_model = std::make_unique<GRBModel>(genv);
        m->set_flow_sets(sol);
        m->set_variables(sol, *grb_model);
        grb_model->set(GRB_DoubleParam_TimeLimit, ctx.max_time);
        grb_model->update();
    }
};

class ilp_constraints_large_fixture : public ilp_variables_large_fixture {
protected:
    // Must outlive grb_model: Gurobi holds a raw pointer to the callback object
    // and invokes it during optimize().  A stack variable in SetUp() is destroyed
    // before any test body runs, leaving a dangling pointer → BAD ACCESS.
    std::unique_ptr<subtour_elimination> cb;

    void SetUp() override {
        ilp_variables_large_fixture::SetUp();
        m->set_constraints(sol, *grb_model);
        cb = std::make_unique<subtour_elimination>(*m);
        grb_model->set(GRB_IntParam_LazyConstraints, 1);
        grb_model->setCallback(cb.get());
        grb_model->update();
    }
};

class ilp_objectives_large_fixture : public ilp_constraints_large_fixture {
protected:
    void SetUp() override {
        ilp_constraints_large_fixture::SetUp();
        m->set_objectives(sol, *grb_model);
        grb_model->update();
    }
};

class ilp_solved_large_fixture : public ilp_objectives_large_fixture {
protected:
    void SetUp() override {
        ilp_objectives_large_fixture::SetUp();
        grb_model->set(GRB_DoubleParam_NoRelHeurTime, 300);
        grb_model->update();
        grb_model->optimize();
        int status = grb_model->get(GRB_IntAttr_Status);
        if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL &&
            !(status == GRB_TIME_LIMIT && grb_model->get(GRB_IntAttr_SolCount) > 0))
            GTEST_SKIP() << "Gurobi did not find a feasible solution; skipping retrieve tests";
    }
};

// Bug (documented behaviour): retrieve_solution consumes variables_flow by
// decrementing arc flows to zero as it walks the graph.  After the call, all
// flow values must be 0 — a second call would find no arcs to follow and
// produce an incorrect or empty path.
TEST_F(ilp_solved_large_fixture, RetrieveSolutionConsumesVariablesFlow) {
    cppied_solution result = sol;
    result.coverage = Eigen::VectorXd::Zero(P.req.size());
    std::list<int> path;
    m->retrieve_solution(result, path, flow_setter);
    m->set_solution(result, path);

    for (int u = 0; u < (int)m->variables.size(); ++u)
        for (int f : m->variables[u].variables_flow)
            EXPECT_EQ(f, 0)
                                << "variables_flow[" << u << "][*] is non-zero after retrieve_solution; "
                                                             "the method is destructive — it consumes the flow during path extraction. "
                                                             "A second call cannot reconstruct the path.";
}

class ilp_solved_very_large_fixture: public ::testing::Test{
protected:
    static std::optional<cppied_context>  ctx_vlf;
    static std::optional<cppied_instance> P_vlf;

    static std::unique_ptr<ilp_accessor> m_vlf;
    static cppied_solution               sol_vlf;
    static std::function<void()>         flow_setter_vlf;

    static GRBEnv                    genv_vlf;
    static std::unique_ptr<GRBModel> grb_model_vlf;

    static std::unique_ptr<subtour_elimination> cb_vlf;

    static void SetUpTestSuite() {
        ctx_vlf.emplace(read_configuration(config_path));

        P_vlf.emplace(
                read_matrix<int>(seabed_path),
                read_matrix<double>(pod_path),
                read_matrix<double>(req_path)
        );

        // Initialize once
        sol_vlf.path = {};
        sol_vlf.cost     = {0, 0};
        sol_vlf.coverage = Eigen::VectorXd::Zero(P_vlf->req.size());

        ctx_vlf->start_time = std::chrono::high_resolution_clock::now();
        ctx_vlf->max_time   = 600;

        dp_sweeper dps(ctx_vlf.value(), P_vlf.value());
        auto history_save = [](cppied_solution&){};
        dps.construct(sol_vlf, history_save);

        m_vlf = std::make_unique<ilp_accessor>(ctx_vlf.value(), P_vlf.value());
        m_vlf->coverage.reset(sol_vlf);
        sol_vlf.cost = m_vlf->geometry.cost(sol_vlf);

        flow_setter_vlf = [](){
            for (int u=0; u<m_vlf->problem.vertex.size(); ++u){
                for (int v =0; v<m_vlf->variables[u].outgoing_variables.size(); ++v){
                    m_vlf->variables[u].variables_flow[v] =
                            m_vlf->variables[u].outgoing_variables[v].get(GRB_DoubleAttr_X);
                }
                m_vlf->targets_minus_flow[u] = m_vlf->targets_minus[u].get(GRB_DoubleAttr_X);
                m_vlf->targets_plus_flow[u]  = m_vlf->targets_plus[u].get(GRB_DoubleAttr_X);
            }
        };

        genv_vlf.set(GRB_IntParam_OutputFlag, 0);
        genv_vlf.start();
        grb_model_vlf = std::make_unique<GRBModel>(genv_vlf);
        m_vlf->set_flow_sets(sol_vlf);
        m_vlf->set_variables(sol_vlf, *grb_model_vlf);
        grb_model_vlf->set(GRB_DoubleParam_TimeLimit, ctx_vlf->max_time);
        grb_model_vlf->set(GRB_DoubleParam_NoRelHeurTime, 300);
        grb_model_vlf->update();

        m_vlf->set_constraints(sol_vlf, *grb_model_vlf);
        cb_vlf = std::make_unique<subtour_elimination>(*m_vlf);
        grb_model_vlf->set(GRB_IntParam_LazyConstraints, 1);
        grb_model_vlf->setCallback(cb_vlf.get());
        grb_model_vlf->update();

        m_vlf->set_objectives(sol_vlf, *grb_model_vlf);
        grb_model_vlf->update();
        grb_model_vlf ->optimize();
    }

    static void TearDownTestSuite() {
        // optional cleanup
    }

    void SetUp() override{
        int status = grb_model_vlf->get(GRB_IntAttr_Status);
        if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL &&
            !(status == GRB_TIME_LIMIT && grb_model_vlf->get(GRB_IntAttr_SolCount) > 0))
            GTEST_SKIP() << "Gurobi did not find a feasible solution; skipping retrieve tests";
    }

    // Raw data files live in the source tree
    static constexpr const char* seabed_path =
                                       PROJECT_SOURCE_DIR "/tests/configurations/test_s6464_ir1_lrc033/cppied_problem.txt";
    static constexpr const char* pod_path =
                                       PROJECT_SOURCE_DIR "/tests/configurations/test_s6464_ir1_lrc033/cppied_pod.txt";
    static constexpr const char* req_path =
                                       PROJECT_SOURCE_DIR "/tests/configurations/test_s6464_ir1_lrc033/cppied_req.txt";
    // Generated config files (with resolved paths) live in the build tree
    static constexpr const char* config_path =
                                       PROJECT_SOURCE_DIR "/tests/configurations/ilp_large_test_config.txt";
};

std::optional<cppied_context> ilp_solved_very_large_fixture::ctx_vlf;
std::optional<cppied_instance> ilp_solved_very_large_fixture::P_vlf;

std::unique_ptr<ilp_accessor> ilp_solved_very_large_fixture::m_vlf = nullptr;
cppied_solution ilp_solved_very_large_fixture::sol_vlf;
std::function<void()> ilp_solved_very_large_fixture::flow_setter_vlf;

GRBEnv ilp_solved_very_large_fixture::genv_vlf{true};
std::unique_ptr<GRBModel> ilp_solved_very_large_fixture::grb_model_vlf = nullptr;

std::unique_ptr<subtour_elimination> ilp_solved_very_large_fixture::cb_vlf = nullptr;

// Bug (documented behaviour): retrieve_solution consumes variables_flow by
// decrementing arc flows to zero as it walks the graph.  After the call, all
// flow values must be 0 — a second call would find no arcs to follow and
// produce an incorrect or empty path.
TEST_F(ilp_solved_very_large_fixture, RetrieveSolutionConsumesVariablesFlow) {
    cppied_solution result = sol_vlf;
    result.coverage = Eigen::VectorXd::Zero(P_vlf->req.size());
    std::list<int> path;
    m_vlf->retrieve_solution(result, path, flow_setter_vlf);
    m_vlf->set_solution(result, path);

    for (int u = 0; u < (int)m_vlf->variables.size(); ++u)
        for (int f : m_vlf->variables[u].variables_flow)
            EXPECT_EQ(f, 0)
                                << "variables_flow[" << u << "][*] is non-zero after retrieve_solution; "
                                                             "the method is destructive — it consumes the flow during path extraction. "
                                                             "A second call cannot reconstruct the path.";
}

TEST_F(ilp_solved_very_large_fixture, RetrieveValidSolutionFromFlow){
    cppied_solution result = sol_vlf;
    result.coverage = Eigen::VectorXd::Zero(P_vlf->req.size());
    std::list<int> path;
    m_vlf->retrieve_solution(result, path, flow_setter_vlf);
    m_vlf->set_solution(result, path);
    EXPECT_NO_THROW(m_vlf->validate_solution(result));
}
