#pragma once

#include "../structures.hpp"
#include "../cppied_instance.hpp"

class path_engine {
public:
    explicit path_engine(const cppied_instance& instance) : problem(instance){
        n_rows = static_cast<int>(problem.seabed.rows());
        n_cols = static_cast<int>(problem.seabed.cols());
        horizontal_bound = (n_rows + 1) * n_cols;

        set_int_coordinates();
        set_boundaries();
        build_dist_table();
    }

    void complete(cppied_solution& pSol) const;
    void complete(std::vector<segment>& path) const;
    [[nodiscard]] cost_t dist(const segment& a,
                const segment& b) const;
    [[nodiscard]] cost_t dist_arithmetic(const segment& a,
                                       const segment& b) const;
    [[nodiscard]] cost_t dist_boundary(const int block,
                                         int dx, int dy) const;
    bool is_true_boundary(const int block,
                          int v, int u, int sdx, int sdy) const;
    [[nodiscard]] cost_t cost(const segment& a) const;
    [[nodiscard]] cost_t cost(const cppied_solution&) const;
    void extend(segment& source,
                     const segment& target) const;
    [[nodiscard]] segment turn(const segment& source,
                 const segment& target) const;
    void link(std::vector<segment>& path, segment& target) const;
    [[nodiscard]] direction get_direction(const segment&) const;
    bool is_in_rectangle(const segment& s, iRectangle& box) const;
    [[nodiscard]] int get_row(const segment&) const;
    [[nodiscard]] int get_column(const segment&) const;
    [[nodiscard]] std::vector<std::pair<int,cost_t>> get_top_k_insertion(const cppied_solution& pSol,
                                             const segment& pSeg,
                                             int k) const;
    [[nodiscard]] bool satisfy(const segment& a, const segment& b) const;
    [[nodiscard]] bool is_horizontal(const segment& a) const;
    void correct_segment_direction(segment& a, direction d);
    segment flip_segment(const segment& s);

    using sVecIt = std::vector<segment>::const_iterator;
    [[nodiscard]] cost_t removal_gain(const cppied_solution& pSol, sVecIt seg) const;
    [[nodiscard]] inline cost_t insert_front_gain(const segment& node, const segment& next) const {
        return cost(node) + dist(node,next);
    }
    [[nodiscard]] inline cost_t insert_between_gain(const segment& prev,
                                                    const segment& node,
                                                    const segment& next) const {
        return dist(prev, node) +
                cost(node) +
                dist(node,next) -
                dist(prev, next);
    }
    [[nodiscard]] inline cost_t insert_tail_gain(const segment& prev, const segment& node) const {
        return dist(prev, node) + cost(node);
    }

    static inline bool is_node(int x) {
        return (unsigned)(x + 2) > 1;
    }

    static constexpr int NULL_NODE          = -1;
    static constexpr int REVERSED_NULL_NODE = -2;

    int n_rows;
    int n_cols;
    int horizontal_bound;
protected:
    const cppied_instance& problem;

    // True only for horizontal nodes on a grid edge (col==0, col==n_cols-1,
    // row==0, row==n_rows).  Vertical nodes are never flagged: no V→V or
    // same-type formula has an absolute-position check on the V-node itself.
    std::vector<uint8_t> boundary_displacement;
    std::vector<cost_t>  displacement_table;

    std::vector<int> vx;
    std::vector<int> vy;

    // dist_table has 8 * H * W entries instead of 16
    // primary_block[from*4+to] → which of the 8 stored blocks to use
    // flip_disp[from*4+to]     → whether to negate (sdx, sdy) before indexing

    //                              EE  EW  ES  EN
    //                              WE  WW  WS  WN
    //                              SE  SW  SS  SN
    //                              NE  NW  NS  NN
    static constexpr int  primary[16] = {0, 1, 2, 3,
                                         1, 0, 4, 5,
                                         5, 3, 6, 7,
                                         4, 2, 7, 6};
    static constexpr bool flip[16]    = {0, 0, 0, 0,
                                         1, 1, 0, 0,
                                         1, 1, 0, 0,
                                         1, 1, 1, 1};

    static constexpr int secondary[16] = {0, 1, 2, 3,
                                          4, 0, 5, 6,
                                          6, 3, 7, 8,
                                          5, 2, 9, 7};  // NE: 6→5 (WS), NW: 3→2 (ES)

    static constexpr bool exchange[16] = {0, 0, 0, 0,
                                          0, 1, 0, 0,   // WW: 0→1
                                          1, 1, 0, 0,
                                          1, 1, 0, 0};  // NN: 0→1

    void set_boundaries();
    void build_dist_table();
    void set_int_coordinates();

    cost_t ee_dist(int,int) const;
    cost_t ew_dist(int,int) const;
    cost_t es_dist(int,int) const;
    cost_t en_dist(int,int) const;

    cost_t ww_dist(int,int) const;
    cost_t we_dist(int,int) const;
    cost_t ws_dist(int,int) const;
    cost_t wn_dist(int,int) const;

    cost_t ss_dist(int,int) const;
    cost_t sn_dist(int,int) const;
    cost_t se_dist(int,int) const;
    cost_t sw_dist(int,int) const;

    cost_t nn_dist(int,int) const;
    cost_t ns_dist(int,int) const;
    cost_t ne_dist(int,int) const;
    cost_t nw_dist(int,int) const;

    cost_t ew_dist_boundary() const;
    cost_t es_dist_boundary(int,int) const;
    cost_t en_dist_boundary(int,int) const;

    cost_t we_dist_boundary() const;
    cost_t ws_dist_boundary(int,int) const;
    cost_t wn_dist_boundary(int,int) const;

    cost_t sn_dist_boundary() const;
    cost_t ns_dist_boundary() const;

    bool ew_is_boundary(int a, int b) const;
    bool es_is_boundary(int,int, int, int) const;
    bool en_is_boundary(int,int, int, int) const;

    bool we_is_boundary(int a, int b) const;
    bool ws_is_boundary(int,int, int, int) const;
    bool wn_is_boundary(int,int, int, int) const;

    bool sn_is_boundary(int a, int b) const;
    bool ns_is_boundary(int a, int b) const;

    static cost_t s_shaped_dist(int dx,int dy,bool backward,bool straight);
};