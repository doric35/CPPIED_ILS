#include "../test_fixture.hpp"

// Exposes protected boundary_displacement for white-box testing of set_boundaries().
struct testable_path_engine : public path_engine {
    explicit testable_path_engine(const cppied_instance& inst) : path_engine(inst) {}
    bool is_boundary(int block, int v) const {
        return boundary_displacement[block * static_cast<int>(problem.vertex.size()) + v];
    }
    std::size_t boundary_size() const { return boundary_displacement.size(); }
};

TEST_F(cppied_method_fixture, ValidRemovalGain){
    cppied_solution sol{
            {{6, 11},
             {23,18},
             {44,42},
             {48, 51},
             {30, 35},
             {75,73},
             {62, 63}},
     Eigen::VectorXd::Zero(P.req.size()),
     cost_t{0,0}
    };
    method_instance.coverage.reset(sol);
    sol.cost = method_instance.geometry.cost(sol);
    EXPECT_EQ(method_instance.geometry.removal_gain(sol,
                                                    std::next(sol.path.begin(),5)),
              (cost_t{2,2}));

    sol.path = {{6, 11},
                {23,18},
                {44,42},
                {48, 51},
                {30, 33},
                {77, 76},
                {75,73},
                {62, 63}};
    sol.cost = method_instance.geometry.cost(sol);
    EXPECT_EQ(method_instance.geometry.removal_gain(sol,
                                                    std::next(sol.path.begin(),4)),
              (cost_t{2,4}));

}

TEST_F(cppied_method_fixture, ValidInsertionGain){
    cppied_solution sol{
            {{6, 11},
             {79, path_engine::NULL_NODE},
             {23,18},
             {44,42},
             {48, 51},
             {30, 35},
             {75,74},
             {10, 9},
             {62, 63}},
            Eigen::VectorXd::Zero(P.req.size()),
            cost_t{0,0}
    };
    method_instance.coverage.reset(sol);
    sol.cost = method_instance.geometry.cost(sol);
    EXPECT_EQ(method_instance.geometry.insert_between_gain(sol.path[6],
                                                           {16,15},
                                                           sol.path[8]),
              (cost_t{0,0}));
}

TEST_F(cppied_method_fixture, ValidTopInsertion){
    cppied_solution sol{
            {{6, 11},
             {23,18},
             {44,42},
             {48, 51},
             {30, 35},
             {75,74},
             //{73, path_engine::REVERSED_NULL_NODE},
             {62, 63}},
            Eigen::VectorXd::Zero(P.req.size()),
            cost_t{0,0}
    };
    method_instance.coverage.reset(sol);
    sol.cost = method_instance.geometry.cost(sol);
    segment s = {16, path_engine::REVERSED_NULL_NODE};
    auto response =
            method_instance.geometry.get_top_k_insertion(sol, s, 1);
    EXPECT_EQ(response[0].second, (cost_t{0,0}));
    EXPECT_EQ(response[0].first, 6);

    response = method_instance.geometry.get_top_k_insertion(sol, s, 2);
    EXPECT_EQ(response[0].second, (cost_t{0,0}));
    EXPECT_EQ(response[0].first, 6);
}

TEST_F(cppied_method_fixture, ValidCompletions){
    int n_nodes = int(P.vertex.size());
    cppied_solution sol{};
    for (int i=0; i< n_nodes; i++){
        segment s_i = {i, path_engine::NULL_NODE};
        segment s_ir = {i, path_engine::REVERSED_NULL_NODE};
        for (int j=0; j< n_nodes; j++){
            segment s_j = {j, path_engine::NULL_NODE};
            segment s_jr = {j, path_engine::REVERSED_NULL_NODE};
            cost_t d_ij = method_instance.geometry.dist(s_i, s_j);
            std::vector<segment> path_ij{s_i, s_j};
            EXPECT_NO_THROW(method_instance.geometry.complete(path_ij)) <<
                "Throwed with " << s_i << " " << s_j;
            sol.path = path_ij;
            EXPECT_EQ(method_instance.geometry.cost(sol), d_ij);
            cost_t d_ijr = method_instance.geometry.dist(s_i, s_jr);
            std::vector<segment> path_ijr{s_i, s_jr};
            EXPECT_NO_THROW(method_instance.geometry.complete(path_ijr)) <<
            "Throwed with " << s_i << " " << s_jr;
            sol.path = path_ijr;
            EXPECT_EQ(method_instance.geometry.cost(sol), d_ijr);
            cost_t d_irj = method_instance.geometry.dist(s_ir, s_j);
            std::vector<segment> path_irj{s_ir, s_j};
            EXPECT_NO_THROW(method_instance.geometry.complete(path_irj)) <<
                "Throwed with " << s_ir << " " << s_j;
            sol.path = path_irj;
            EXPECT_EQ(method_instance.geometry.cost(sol), d_irj);
            cost_t d_irjr = method_instance.geometry.dist(s_ir, s_jr);
            std::vector<segment> path_irjr{s_ir, s_jr};
            EXPECT_NO_THROW(method_instance.geometry.complete(path_irjr)) <<
                "Throwed with " << s_ir << " " << s_jr;;
            sol.path = path_irjr;
            EXPECT_EQ(method_instance.geometry.cost(sol), d_irjr);
        }
    }
}

TEST_F(cppied_method_fixture, ValidDistancesComputations){
    segment s1{66, -1}, s2{66, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={66, -1}, s2={66, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,4}));

    s1={63, -1}, s2 = {63, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={63, -1}, s2={63, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,4}));

    s1={63, -1}, s2={63, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,4}));

    s1={26, path_engine::REVERSED_NULL_NODE}, s2={63, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,3}));

    s1={57, path_engine::REVERSED_NULL_NODE}, s2={63, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{2,2}));

    s1={20, path_engine::NULL_NODE}, s2={63, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{1,1}));

    s1={63, 64}, s2={63, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={27, -2}, s2={27, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,4}));

    s1={27, 26}, s2={27, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={5, -1}, s2={5, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{7,6}));

    s1={78, -1}, s2={5, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{6,5}));

    s1={11, -2}, s2={5, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={73, -1}, s2={5, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,3}));

    s1={17, -1}, s2={5, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,2}));

    s1={79, -2}, s2={5, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{2,1}));

    s1={36, -1}, s2={36, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={37, -1}, s2={36, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,4}));

    s1={59, -2}, s2={36, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,3}));

    s1={31, -2}, s2={36, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{2,2}));

    s1={0, path_engine::REVERSED_NULL_NODE}, s2={48, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{6,5}));

    s1={42, path_engine::NULL_NODE}, s2={48, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={6, path_engine::NULL_NODE}, s2={48, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,3}));

    s1={7, path_engine::NULL_NODE}, s2={48, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,3}));

    s1={0, path_engine::NULL_NODE}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{6,5}));

    s1={48, path_engine::NULL_NODE}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={7, path_engine::NULL_NODE}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,3}));

    s1={54, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,2}));

    s1={1, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{2,1}));

    s1={0, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{1,1}));

    s1={0, 1}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,5}));

    s1={54, -1}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,4}));

    s1={6, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{6,5}));

    s1={43, -1}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={12, -1}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,3}));

    s1={49, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,2}));

    s1={48, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{2,2}));

    s1={0, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{1,1}));

    s1={42, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{7,6}));

    s1={6, -1}, s2={43, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,3}));

    s1={48, -2}, s2={43, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,2}));

    s1={0, -2}, s2={43, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{2,1}));

    s1={42, -1}, s2={43, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{1,0}));

    s1={6, 7}, s2={43, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,3}));

    s1={6, -2}, s2={55, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,3}));

    s1={36, -1}, s2={30, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={53, -2}, s2={30, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,3}));

    s1={52, -2}, s2={30, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,3}));

    s1={24, -2}, s2={30, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{2,2}));

    s1={46, -1}, s2={30, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{1,1}));

    s1={36, 37}, s2={30, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{6,4}));

    s1={36, -1}, s2={47, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{6,5}));

    s1={36, -2}, s2={47, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{6,5}));

    s1={47, -2}, s2={47, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={30, -1}, s2={47, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,3}));

    s1={52, -2}, s2={47, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,2}));

    s1={42, -2}, s2={50, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{7,4}));

    s1 = {30, 35}, s2 = {75,73};
    EXPECT_EQ(method_instance.geometry.dist(s1,s2), (cost_t{3,3}));

    s1 = {75,73}, s2 = {62, 63};
    EXPECT_EQ(method_instance.geometry.dist(s1,s2), (cost_t{4,2}));

    s1 = {75,74}, s2 = {10, 9};
    EXPECT_EQ(method_instance.geometry.dist(s1,s2), (cost_t{2,1}));

    s1 = {10,9}, s2 = {62, 63};
    EXPECT_EQ(method_instance.geometry.dist(s1,s2), (cost_t{2,1}));

    s1 = {75, 74}, s2 = {62, 63};
    EXPECT_EQ(method_instance.geometry.dist(s1,s2), (cost_t{3,2}));

    s1 = {75,74}, s2 = {16, 15};
    EXPECT_EQ(method_instance.geometry.dist(s1,s2), (cost_t{1,1}));

    s1 = {16,15}, s2 = {62, 63};
    EXPECT_EQ(method_instance.geometry.dist(s1,s2), (cost_t{1,1}));

    s1 = {0, -1}, s2 = {6, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1,s2), (cost_t{5,4}));

    s1 = {48, -1}, s2 = {6, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1,s2), (cost_t{4,3}));

    s1 = {49, -1}, s2 = {6, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1,s2), (cost_t{3,3}));

    s1 = {12, -2}, s2 = {6, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1,s2), (cost_t{2,2}));

    s1 = {43, -2}, s2 = {6, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1,s2), (cost_t{1,1}));

    s1 = {36, -2}, s2 = {53, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1,s2), (cost_t{6,5}));

}

TEST_F(cppied_method_fixture, SpecificPathCompletions){
    segment s_i = {63, path_engine::NULL_NODE};
    segment s_j = {63, path_engine::NULL_NODE};
    std::vector<segment> path_ij{s_i, s_j};
    EXPECT_NO_THROW(method_instance.geometry.complete(path_ij)) <<
        "Throwed with " << s_i << " " << s_j;

    s_i = {36, path_engine::NULL_NODE}, s_j = {36, path_engine::REVERSED_NULL_NODE};
    path_ij = {s_i, s_j};
    EXPECT_NO_THROW(method_instance.geometry.complete(path_ij)) <<
        "Throwed with " << s_i << " " << s_j;

    s_i = {0, path_engine::NULL_NODE}, s_j = {48, path_engine::REVERSED_NULL_NODE};
    path_ij = {s_i, s_j};
    EXPECT_NO_THROW(method_instance.geometry.complete(path_ij)) <<
        "Throwed with " << s_i << " " << s_j;

    s_i = {0, path_engine::NULL_NODE}, s_j = {42, path_engine::NULL_NODE};
    path_ij = {s_i, s_j};
    EXPECT_NO_THROW(method_instance.geometry.complete(path_ij)) <<
        "Throwed with " << s_i << " " << s_j;

    s_i = {6, -2}, s_j = {42, -1};
    path_ij = {s_i, s_j};
    EXPECT_NO_THROW(method_instance.geometry.complete(path_ij)) <<
        "Throwed with " << s_i << " " << s_j;

    s_i = {6, -1}, s_j = {43, -1};
    path_ij = {s_i, s_j};
    EXPECT_NO_THROW(method_instance.geometry.complete(path_ij)) <<
        "Throwed with " << s_i << " " << s_j;
}

// White-box tests for set_boundaries().
//
// Grid layout for the 6×6 test seabed (n_rows=6, n_cols=6):
//   horizontal nodes: index = row*6 + col   (rows 0–6, cols 0–5  → 0–41)
//   vertical   nodes: index = 42 + col*6+row (cols 0–6, rows 0–5  → 42–83)
//   horizontal_bound = 42, S = 84
//
// The secondary[] table maps each of the 16 direction-pair combos to one of
// 10 canonical blocks (0–9).  set_boundaries() flags nodes in those blocks
// where the fast displacement-table formula would be wrong, causing dist()
// to fall back to the exact dist_boundary() formula.
//
// Block → direction pair(s):
//   0: EE/WW (nothing flagged – formula always valid)
//   1: EW    (rightmost horizontal column)
//   2: ES/NW (top horizontal row + right horizontal column + left vertical column + bottom-right vertical corner)
//   3: EN/SW (bottom horizontal row + right horizontal column + right vertical column + bottom-left vertical corner)
//   4: WE    (leftmost horizontal column)
//   5: WS/NE (top horizontal row + left horizontal column + top-left vertical triangle)
//   6: WN/SE (bottom horizontal row + left horizontal column + top-right vertical staircase + bottom-left vertical staircase)
//   7: SS/NN (nothing flagged)
//   8: SN    (last-row vertical nodes per column)
//   9: NS    (first-row vertical nodes per column)
TEST_F(cppied_method_fixture, ValidSetBoundaries) {
    testable_path_engine eng(P);
    const int S = static_cast<int>(P.vertex.size());  // 84

    // ── size ──────────────────────────────────────────────────────────────
    EXPECT_EQ(eng.boundary_size(), static_cast<std::size_t>(10 * S));

    // ── Block 0 (EE / WW): nothing flagged ────────────────────────────────
    for (int v = 0; v < S; v++)
        EXPECT_FALSE(eng.is_boundary(0, v))
            << "Block 0 (EE): unexpected boundary at v=" << v;

    // ── Block 1 (EW): rightmost column of horizontal nodes ────────────────
    // Expected: v = 5, 11, 17, 23, 29, 35, 41  (col 5 of each horizontal row)
    for (int v = eng.n_cols - 1; v < eng.horizontal_bound; v += eng.n_cols)
        EXPECT_TRUE(eng.is_boundary(1, v))
            << "Block 1 (EW): expected boundary at v=" << v;
    // Left column must NOT be flagged
    for (int v = 0; v < eng.horizontal_bound; v += eng.n_cols)
        EXPECT_FALSE(eng.is_boundary(1, v))
            << "Block 1 (EW): unexpected boundary at v=" << v;
    // Interior horizontal node (row 1, col 3 → v=9)
    EXPECT_FALSE(eng.is_boundary(1, 9));

    // ── Block 2 (ES / NW) ─────────────────────────────────────────────────
    // Top horizontal row (row 0, cols 0–5) must be flagged
    for (int v = 0; v < eng.n_cols; v++)
        EXPECT_TRUE(eng.is_boundary(2, v))
            << "Block 2 (ES): expected boundary at top-row v=" << v;
    // Right horizontal column must be flagged
    for (int v = eng.n_cols - 1; v < eng.horizontal_bound; v += eng.n_cols)
        EXPECT_TRUE(eng.is_boundary(2, v))
            << "Block 2 (ES): expected boundary at right-col v=" << v;
    // First vertical node of each column (col 0 top → v=42; col 1 top → v=48 …)
    for (int c = 0; c < eng.n_cols; c++)
        EXPECT_TRUE(eng.is_boundary(2, eng.horizontal_bound + c * eng.n_rows))
            << "Block 2 (ES): expected boundary at vertical top c=" << c;
    // Interior horizontal node should NOT be flagged (row 2, col 1 → v=13)
    EXPECT_FALSE(eng.is_boundary(2, 13));
    // Interior vertical node should NOT be flagged (col 1, row 2 → v=50)
    EXPECT_FALSE(eng.is_boundary(2, 50));

    // ── Block 3 (EN / SW) ─────────────────────────────────────────────────
    // Bottom horizontal row (row n_rows = 6, cols 0–5 → v=36..41) must be flagged
    for (int v = eng.horizontal_bound - eng.n_cols; v < eng.horizontal_bound; v++)
        EXPECT_TRUE(eng.is_boundary(3, v))
            << "Block 3 (EN): expected boundary at bottom-row v=" << v;
    // Right horizontal column must be flagged
    for (int v = eng.n_cols - 1; v < eng.horizontal_bound; v += eng.n_cols)
        EXPECT_TRUE(eng.is_boundary(3, v))
            << "Block 3 (EN): expected boundary at right-col v=" << v;
    // Last vertical node of each column (bottom row of col c → v=47,53,...,77)
    for (int c = 0; c < eng.n_cols; c++)
        EXPECT_TRUE(eng.is_boundary(3, eng.horizontal_bound + c * eng.n_rows + (eng.n_rows - 1)))
            << "Block 3 (EN): expected boundary at vertical bottom c=" << c;

    //Last vertical node of each row must be a boundary
    for (int v = S - 6; v< S; v++){
        EXPECT_TRUE(eng.is_boundary(3, v))
                            << "Block 3 (EN): expected boundary at vertical right v=" << v;
    }
    // Top-left horizontal node (row 0, col 0 → v=0) must NOT be flagged
    EXPECT_FALSE(eng.is_boundary(3, 0));
    // Interior horizontal node (row 2, col 1 → v=13) must NOT be flagged
    EXPECT_FALSE(eng.is_boundary(3, 13));
    // Interior vertical node (col 1, row 2 → v=50) must NOT be flagged
    EXPECT_FALSE(eng.is_boundary(3, 50));

    // ── Block 4 (WE): leftmost column of horizontal nodes ─────────────────
    // Expected: v = 0, 6, 12, 18, 24, 30, 36
    for (int v = 0; v < eng.horizontal_bound; v += eng.n_cols)
        EXPECT_TRUE(eng.is_boundary(4, v))
            << "Block 4 (WE): expected boundary at v=" << v;
    // Right column must NOT be flagged
    for (int v = eng.n_cols - 1; v < eng.horizontal_bound; v += eng.n_cols)
        EXPECT_FALSE(eng.is_boundary(4, v))
            << "Block 4 (WE): unexpected boundary at v=" << v;
    // Interior horizontal node (row 1, col 3 → v=9) must NOT be flagged
    EXPECT_FALSE(eng.is_boundary(4, 9));

    // ── Block 5 (WS / NE) ─────────────────────────────────────────────────
    // Top horizontal row (row 0) must be flagged
    for (int v = 0; v < eng.n_cols; v++)
        EXPECT_TRUE(eng.is_boundary(5, v))
            << "Block 5 (WS): expected boundary at top-row v=" << v;
    // Left horizontal column must be flagged
    for (int v = 0; v < eng.horizontal_bound; v += eng.n_cols)
        EXPECT_TRUE(eng.is_boundary(5, v))
            << "Block 5 (WS): expected boundary at left-col v=" << v;
    // Right-column horizontal nodes must NOT be flagged (except first)
    for (int v = (2*eng.n_cols) - 1; v < eng.horizontal_bound; v += eng.n_cols)
        EXPECT_FALSE(eng.is_boundary(5, v))
            << "Block 5 (WS): unexpected boundary at right-col v=" << v;
    // Interior horizontal node (row 2, col 3 → v=15) must NOT be flagged
    EXPECT_FALSE(eng.is_boundary(5, 15));

    // ── Block 6 (WN / SE) ─────────────────────────────────────────────────
    // Bottom horizontal row must be flagged
    for (int v = eng.horizontal_bound - eng.n_cols; v < eng.horizontal_bound; v++)
        EXPECT_TRUE(eng.is_boundary(6, v))
            << "Block 6 (WN): expected boundary at bottom-row v=" << v;
    // Left horizontal column must be flagged
    for (int v = 0; v < eng.horizontal_bound; v += eng.n_cols)
        EXPECT_TRUE(eng.is_boundary(6, v))
            << "Block 6 (WN): expected boundary at left-col v=" << v;
    // Right-column nodes except bottom must NOT be flagged (e.g. v=5, row 0, col 5)
    EXPECT_FALSE(eng.is_boundary(6, 5));
    // Interior horizontal node (row 2, col 3 → v=15) must NOT be flagged
    EXPECT_FALSE(eng.is_boundary(6, 15));

    // ── Block 7 (SS / NN): nothing flagged ────────────────────────────────
    for (int v = 0; v < S; v++)
        EXPECT_FALSE(eng.is_boundary(7, v))
            << "Block 7 (SS): unexpected boundary at v=" << v;

    // ── Block 8 (SN): last-row vertical node of each column ───────────────
    // Expected: v = 47, 53, 59, 65, 71, 77, 83
    for (int c = 0; c <= eng.n_cols; c++) {
        int v = eng.horizontal_bound + c * eng.n_rows + (eng.n_rows - 1);
        EXPECT_TRUE(eng.is_boundary(8, v))
            << "Block 8 (SN): expected boundary at vertical bottom c=" << c << " v=" << v;
    }
    // First-row vertical nodes must NOT be flagged
    for (int c = 0; c <= eng.n_cols; c++) {
        int v = eng.horizontal_bound + c * eng.n_rows;
        EXPECT_FALSE(eng.is_boundary(8, v))
            << "Block 8 (SN): unexpected boundary at vertical top c=" << c << " v=" << v;
    }

    // ── Block 9 (NS): first-row vertical node of each column ──────────────
    // Expected: v = 42, 48, 54, 60, 66, 72, 78
    for (int c = 0; c <= eng.n_cols; c++) {
        int v = eng.horizontal_bound + c * eng.n_rows;
        EXPECT_TRUE(eng.is_boundary(9, v))
            << "Block 9 (NS): expected boundary at vertical top c=" << c << " v=" << v;
    }
    // Last-row vertical nodes must NOT be flagged
    for (int c = 0; c <= eng.n_cols; c++) {
        int v = eng.horizontal_bound + c * eng.n_rows + (eng.n_rows - 1);
        EXPECT_FALSE(eng.is_boundary(9, v))
            << "Block 9 (NS): unexpected boundary at vertical bottom c=" << c << " v=" << v;
    }
}