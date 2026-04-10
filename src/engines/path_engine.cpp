#include "../../include/engines/path_engine.hpp"

cost_t path_engine::dist_boundary(const segment &a, const segment &b) const {
    const int from = static_cast<int>(get_direction(a));
    const int to   = static_cast<int>(get_direction(b));
    const int v    = is_node(a.target) ? a.target : a.source;
    const int u    = b.source;

    switch (from << 2 | to) {          // 0–15, dense, compiler builds a jump table
        case 0:  return ee_dist(v, u);
        case 1:  return ew_dist(v, u);
        case 2:  return es_dist(v, u);
        case 3:  return en_dist(v, u);
        case 4:  return we_dist(v, u);
        case 5:  return ww_dist(v, u);
        case 6:  return ws_dist(v, u);
        case 7:  return wn_dist(v, u);
        case 8:  return se_dist(v, u);
        case 9:  return sw_dist(v, u);
        case 10: return ss_dist(v, u);
        case 11: return sn_dist(v, u);
        case 12: return ne_dist(v, u);
        case 13: return nw_dist(v, u);
        case 14: return ns_dist(v, u);
        case 15: return nn_dist(v, u);
        default: __builtin_unreachable();
    }
}

cost_t path_engine::dist(const segment &a,
                         const segment &b) const {
    const int from = static_cast<int>(get_direction(a));
    const int to = static_cast<int>(get_direction(b));
    const int v = is_node(a.target) ? a.target : a.source;
    const int u = b.source;

    const int W = 2*n_cols + 1, H = 2*n_rows + 1;

    if (!boundary_node[v] && !boundary_node[u]) [[likely]] {
        const int block = from * 4 + to;
        int sdx = vx[u] - vx[v];
        int sdy = vy[u] - vy[v];
        if (flip[block]) sdx = -sdx, sdy = -sdy;
        return displacement_table[(primary[block] * H * W) + ((sdy + n_rows) * W) + (sdx + n_cols)];
    }
    return dist_boundary(a, b);
}

void path_engine::set_boundaries() {
    boundary_node.resize(problem.vertex.size(), false);
    for (int v=0; v<horizontal_bound; v++){
        const int col = v % n_cols, row = v / n_cols;
        if (col == 0 || col == n_cols-1 || row == 0 || row == n_rows)
            boundary_node[v] = true;
    }
    for (int v=horizontal_bound; v<(int)problem.vertex.size(); v++){
        const int local = v - horizontal_bound;
        const int col = local / n_rows, row = local % n_rows;
        if (col == 0 || col == n_cols || row == 0 || row == n_rows - 1)
            boundary_node[v] = true;
    }
}

void path_engine::set_int_coordinates() {
    vx.resize(problem.vertex.size());
    vy.resize(problem.vertex.size());
    for (int v =0; v < (int)problem.vertex.size(); v++){
        vx[v] = static_cast<int>(problem.vertex[v](0));
        vy[v] = static_cast<int>(problem.vertex[v](1));
    }
}

void path_engine::set_displacements() {
    constexpr int B_EE = 0, B_EW = 1, B_ES = 2, B_EN = 3;
    constexpr int B_WS = 4, B_WN = 5, B_SS = 6, B_SN = 7;

    const int W = 2*n_cols + 1, H = 2*n_rows + 1;
    displacement_table.resize(8*W*H);

    auto write = [&](int block, int sdx, int sdy, cost_t c) {
        displacement_table[(block * W*H) + ((sdy + n_rows) * W) + (sdx + n_cols)] = c;
    };

    const int h_ref     = n_cols + 1;                         // horizontal: col=1, row=1
    const int h_ref_bot = (n_rows - 1) * n_cols + 1;          // horizontal: col=1, row=n_rows-1
    const int v_ref     = horizontal_bound + n_rows + 1;       // vertical:   col=1, row=1
    const int v_ref_bot = horizontal_bound + 2 * n_rows - 2;   // vertical:   col=1, row=n_rows-2

    // ── Horizontal → Horizontal (EE, EW) ────────────────────────────────────────
    // Two sources cover all four (sdx, sdy) quadrants:
    //   h_ref (top-left):    forward=(+,+), rotation=(−,−)
    //   h_ref_bot (bot-left): forward=(+,−), rotation=(−,+)
    for (int src : {h_ref, h_ref_bot}) {
        for (int v = h_ref; v < horizontal_bound; ++v) {
            if (boundary_node[v]) continue;
            const int sdx = vx[v] - vx[src];
            const int sdy = vy[v] - vy[src];
            write(B_EE,  sdx,  sdy, ee_dist(src, v));
            write(B_EE, -sdx, -sdy, ee_dist(v, src));
            write(B_EW,  sdx,  sdy, ew_dist(src, v));
            write(B_EW, -sdx, -sdy, ew_dist(v, src));
        }
    }

    // ── Horizontal → Vertical (ES, EN, WS, WN) ──────────────────────────────────
    // Four loops covering all quadrants:
    //   Loop A: h_ref as source,     all vertical targets   → (+,+)
    //   Loop B: all horizontal srcs, v_ref as target        → (−,−)
    //   Loop C: h_ref_bot as source, all vertical targets   → (+,−)
    //   Loop D: all horizontal srcs, v_ref_bot as target    → (−,+)
    auto write_cross = [&](int src_h, int tgt_v) {
        const int sdx = vx[tgt_v] - vx[src_h];
        const int sdy = vy[tgt_v] - vy[src_h];
        write(B_ES, sdx, sdy, es_dist(src_h, tgt_v));
        write(B_EN, sdx, sdy, en_dist(src_h, tgt_v));
        write(B_WS, sdx, sdy, ws_dist(src_h, tgt_v));
        write(B_WN, sdx, sdy, wn_dist(src_h, tgt_v));
    };
    // Loops A and C: fixed horizontal source, all vertical targets
    for (int src : {h_ref, h_ref_bot}) {
        for (int v = v_ref; v < (int)problem.vertex.size(); ++v) {
            if (boundary_node[v]) continue;
            write_cross(src, v);
        }
    }
    // Loops B and D: all horizontal sources, fixed vertical target
    for (int tgt : {v_ref, v_ref_bot}) {
        for (int v = h_ref; v < horizontal_bound; ++v) {
            if (boundary_node[v]) continue;
            write_cross(v, tgt);
        }
    }

    // ── Vertical → Vertical (SS, SN) ────────────────────────────────────────────
    // Same two-source strategy as horizontal → horizontal.
    for (int src : {v_ref, v_ref_bot}) {
        for (int v = v_ref; v < (int)problem.vertex.size(); ++v) {
            if (boundary_node[v]) continue;
            const int sdx = vx[v] - vx[src];
            const int sdy = vy[v] - vy[src];
            write(B_SS,  sdx,  sdy, ss_dist(src, v));
            write(B_SS, -sdx, -sdy, ss_dist(v, src));
            write(B_SN,  sdx,  sdy, sn_dist(src, v));
            write(B_SN, -sdx, -sdy, sn_dist(v, src));
        }
    }
}

bool path_engine::satisfy(const segment& a, const segment& b) const {
    return dist(a,b) <= cost_t{1,1};
}

cost_t path_engine::cost(const segment &a) const {
    if (is_node(a.target))
        return {std::abs(a.target - a.source), 0};
    return {0,0};
}

cost_t path_engine::cost(const cppied_solution &pSol) const {
    cost_t c = cost(pSol.path.front());
    auto func = [&](
            const segment& prev, const segment& node, cost_t& acc){
        acc += dist(prev, node) + cost(node);
    };
    auto it = std::next(pSol.path.begin());
    for (; it!= pSol.path.end(); ++it)
        func(*std::prev(it), *it, c);
    return c;
}

cost_t path_engine::removal_gain(const cppied_solution &pSol, path_engine::sVecIt seg) const {
    cost_t gain{0,0};
    if (seg != pSol.path.cbegin())
        gain += dist(*std::prev(seg), *seg);
    gain += cost(*seg);

    if (seg != std::prev(pSol.path.cend())) {
        gain += dist(*seg, *std::next(seg));
    }

    if (seg != pSol.path.cbegin()
        && seg != std::prev(pSol.path.cend())){
        gain -= dist(*std::prev(seg), *std::next(seg));
    }

    return gain;
}

bool path_engine::is_horizontal(const segment &a) const {
    direction dir = get_direction(a);
    return dir == direction::E || dir == direction::W;
}

bool path_engine::is_in_rectangle(const segment &s, iRectangle &box) const {
    bool in = problem.vertex[s.source](0) >= box.ul.x;
    in = in && problem.vertex[s.source](1) >= box.ul.y;
    in = in && problem.vertex[s.source](0) <= box.lr.x;
    in = in && problem.vertex[s.source](1) <= box.lr.y;
    if (is_node(s.target)){
        in = in && problem.vertex[s.target](0) >= box.ul.x;
        in = in && problem.vertex[s.target](1) >= box.ul.y;
        in = in && problem.vertex[s.target](0) <= box.lr.x;
        in = in && problem.vertex[s.target](1) <= box.lr.y;
    }

    return in;
}

void path_engine::correct_segment_direction(segment &a, direction d) {
    direction current = get_direction(a);
    if (current != d){
        if (is_node(a.target))
            a = {a.target, a.source};
        else{
            a = a.target == NULL_NODE ?
                    segment{a.source, REVERSED_NULL_NODE} : segment{a.source, NULL_NODE};
        }
    }
}

direction path_engine::get_direction(const segment &seg) const {
    int dir = seg.source >= horizontal_bound;
    dir = dir << 1;
    int reversed = (seg.target < seg.source);
    reversed = reversed >> (seg.target == NULL_NODE);
    dir = dir | reversed;
    return static_cast<direction>(dir);
}

segment path_engine::flip_segment(const segment &s){
    if (!is_node(s.source)) return s;
    direction dir = get_direction(s);
    direction other_dir = flip_direction(dir);
    segment other = s;
    correct_segment_direction(other, other_dir);
    return other;
}

int path_engine::get_row(const segment& seg) const {
    assert(get_direction(seg) == direction::E || get_direction(seg) == direction::W);
    return seg.source / n_cols;
}

int path_engine::get_column(const segment &seg) const {
    assert(get_direction(seg) == direction::S || get_direction(seg) == direction::N);
    return (seg.source - horizontal_bound) / n_rows;
}

void path_engine::complete(cppied_solution &pSolution) const{
    if (pSolution.path.empty())
        return;
    complete(pSolution.path);
}

void path_engine::complete(std::vector<segment>& path) const{
    std::vector<segment> new_path;
    new_path.reserve(2*path.size());
    new_path.push_back(path.front());

    for (int i=1; i<path.size(); ++i)
        link(new_path, path[i]);

    path.swap(new_path);
}

void path_engine::link(std::vector<segment> &path,
                       segment& target) const {
    auto stopping_criterion = [&](
            const segment& a, const segment& b){
        return dist(a, b) == cost_t{1,1};
    };
    while (!stopping_criterion(path.back(), target)){
        extend(path.back(), target);
        if (dist(path.back(), target) <= cost_t{1,0}) {
            path.back().target = is_node(target.target) ?
                                 target.target : target.source;
            return;
        }
        segment new_seg = turn(path.back(),target);
        if (is_node(new_seg.source))
            path.push_back(new_seg);
    }
    if (dist(path.back(), target) <= cost_t{1,0})
        path.back().target = is_node(target.target) ?
                             target.target : target.source;
    else
        path.push_back(target);
}

void path_engine::extend(segment &pSource, const segment &pTarget) const{
    cost_t weight = dist(pSource, pTarget);
    direction edge_dir = get_direction(pSource);
    int dummy = (edge_dir == direction::E || edge_dir == direction::S) ?
            NULL_NODE : REVERSED_NULL_NODE;
    int current = !is_node(pSource.target) ? pSource.source : pSource.target;
    bool dead_end = false;

    while (!dead_end){

        int prev = current;

        for (SMiIt v(problem.adj, current); v; ++v){

            segment trial = {v.index(), dummy};
            if (get_direction(trial) != edge_dir) continue;

            trial = {current, v.index()};
            if (get_direction(trial) != edge_dir) continue;

            cost_t dt = dist(trial, pTarget);

            if (dt < weight) {
                weight = dt;
                current = v.index();
                break;
            }
        }
        dead_end = prev == current;
    }

    pSource.target = current == pSource.source ? pSource.target : current;
}

segment path_engine::turn(const segment &pSource, const segment &pTarget) const{
    direction edge_dir = get_direction(pSource);
    cost_t weight = dist(pSource, pTarget);
    std::array<segment, 2> candidates{};

    if (weight.length <=1)
        return {NULL_NODE,NULL_NODE};

    int current = !is_node(pSource.target) ? pSource.source : pSource.target;

    for (SMiIt v(problem.adj, current); v; ++v){
        candidates[0] = {v.index(), NULL_NODE};
        candidates[1] = {v.index(), REVERSED_NULL_NODE};
        if (get_direction(candidates[0]) == edge_dir ||
            get_direction(candidates[1]) == edge_dir) continue;

        for (auto& edge: candidates){
            cost_t dt1 = dist(pSource,edge);
            if (dt1 == cost_t{1,1}){
                cost_t dt2 = dist(edge, pTarget);
                cost_t triangular_dist = dt1 + dt2;
                if (triangular_dist == weight) return edge;
            }
        }
    }
    throw std::runtime_error("Did not find a valid direction.");
}

std::vector<std::pair<int, cost_t>> path_engine::get_top_k_insertion(
        const cppied_solution& pSol,
        const segment& pTrial,
        int k) const{
    std::vector<int> candidates(pSol.path.size()-1);
    std::iota(candidates.begin(), candidates.end(), 1);
    std::vector<std::pair<int,cost_t>> solution;
    solution.reserve(k);

    auto func = [&](const int & i){
        return std::pair<int,cost_t>{i,
                                     insert_between_gain(pSol.path[i-1], pTrial, pSol.path[i])};
    };
    std::transform(candidates.begin(), candidates.end(),
                   std::back_inserter(solution), func);
    auto cmp = [&](
            const std::pair<int,cost_t> & a,
            const std::pair<int,cost_t> & b){
        return a.second < b.second;
    };

    std::partial_sort(solution.begin(),
                      std::next(solution.begin(),k),
                      solution.end(), cmp);
    std::vector<std::pair<int,cost_t>> top_k(solution.begin(), std::next(solution.begin(), k));
    return top_k;
}

cost_t
path_engine::ee_dist(int a, int b) const
{
    const auto& va = problem.vertex[a];
    const auto& vb = problem.vertex[b];

    //Integer coordinates
    int xa = int(va(0)), ya = int(va(1));
    int xb = int(vb(0)), yb = int(vb(1));

    //Integer distances (valid)
    int dy = std::abs(ya - yb);
    int dx = std::abs(xa - xb);

//    bool fourth_quadrant = (xa < xb) && (ya < yb);
//    bool second_quadrant = (xa < xb) && (ya > yb);
    bool first_quadrant = (xa > xb) && (ya > yb);
    bool third_quadrant = (xa > xb) && (ya < yb);

    //Base turn count
    bool backward = (xb <= xa);     // need to reverse direction
    bool forward = !backward;
    bool straight = forward & (yb == ya);

    //Edge case
    if (backward && dy <= 1)
        return { 4 + dx + dy, 4};

    return s_shaped_dist(dx, dy, backward, straight);
}

cost_t
path_engine::ew_dist(int a, int b) const
{
    //Degen case
    if (a == b && a % n_cols == (n_cols - 1))
        return {7, 6};
    else if (a == b)
        return {5, 4};

    const auto& va = problem.vertex[a];
    const auto& vb = problem.vertex[b];

    //Integer coordinates
    int xa = int(va(0)), ya = int(va(1));
    int xb = int(vb(0)), yb = int(vb(1));

    //Integer distances (valid)
    int dy = std::abs(ya - yb);
    int dx = std::abs(xa - xb);

    //Base turn count
    int n_turns = 2;

    //Base length count;
    int length = n_turns + dx + dy;

    //Geometric corrections
    if (dy == 0) {
        n_turns += 2;
        length += 1;
    } else
        length -= 1;
    return {length, n_turns};
}

cost_t
path_engine::es_dist(int a, int b) const
{
    const auto& va = problem.vertex[a];
    const auto& vb = problem.vertex[b];

    //Coordinates
    double xa = va(0), ya = va(1);
    double xb = vb(0), yb = vb(1);

    //integer distances
    int dy = std::abs(ya - yb);
    int dx = std::abs(xa - xb);

    //Base turn count
    int n_turns = 1;
    bool fourth_quadrant = (xa < xb) && (ya < yb);
    bool second_quadrant = (xa < xb) && (ya > yb);
    //bool first_quadrant = (xa > xb) && (ya > yb);
    bool third_quadrant = (xa > xb) && (ya < yb);

    //Degen case
    if (a % n_cols == n_cols - 1 && second_quadrant)
        return {5 + std::abs(1 - dy), 5};
    else if (a / problem.seabed.cols() == 0 && (dy == 0) && third_quadrant)
        return {5 + std::abs(1 - dx), 5};

    if (!fourth_quadrant) n_turns += 2;

    //Base length count
    int length = n_turns + dx + dy;

    //Geometric corrections
    if ((second_quadrant && dx == 0) || (third_quadrant && dy==0)) length += 1;
    else if (third_quadrant || second_quadrant) length -= 1;

    return {length, n_turns};
}

cost_t
path_engine::en_dist(int a, int b) const
{
    const auto& va = problem.vertex[a];
    const auto& vb = problem.vertex[b];

    //Coordinates
    double xa = va(0), ya = va(1);
    double xb = vb(0), yb = vb(1);

    //integer distances
    int dy = std::abs(ya - yb);
    int dx = std::abs(xa - xb);

    //Base turn count
    int n_turns = 1;
    bool fourth_quadrant = (xa < xb) && (ya < yb);
    bool second_quadrant = (xa < xb) && (ya > yb);
    bool first_quadrant = (xa > xb) && (ya > yb);
    bool third_quadrant = (xa > xb) && (ya < yb);

    //Degen case
    if (a % n_cols == n_cols - 1 && dx == 0 && fourth_quadrant)
        return {5 + std::abs(1 - dy), 5};
    else if (a / n_cols == n_rows && first_quadrant && dy == 0)
        return {5 + std::abs(1 - dx), 5};

    if (!second_quadrant) n_turns += 2;

    //Base length count
    int length = n_turns + dx + dy;

    //Geometric corrections
    if ((fourth_quadrant && dx == 0) || (first_quadrant && dy==0)) length += 1;
    else if (first_quadrant || fourth_quadrant) length -= 1;

    return {length, n_turns};
}

cost_t
path_engine::ww_dist(int a, int b) const
{
    return ee_dist(b,a);
}

cost_t
path_engine::we_dist(int a, int b) const
{
    //Degen case
    if (a == b && a % problem.seabed.cols() == 0)
        return {7, 6};
    else if (a == b)
        return {5, 4};

    //Else flip
    int tmp = a;
    a=b;
    b=tmp;

    const auto& va = problem.vertex[a];
    const auto& vb = problem.vertex[b];

    //Integer coordinates
    int xa = int(va(0)), ya = int(va(1));
    int xb = int(vb(0)), yb = int(vb(1));

    //Integer distances (valid)
    int dy = std::abs(ya - yb);
    int dx = std::abs(xa - xb);

    //Base turn count
    int n_turns = 2;

    //Base length count;
    int length = n_turns + dx + dy;

    //Geometric corrections
    if (dy == 0) {
        n_turns += 2;
        length += 1;
    } else
        length -= 1;
    return {length, n_turns};
}

cost_t
path_engine::ws_dist(int a, int b) const
{
    const auto& va = problem.vertex[a];
    const auto& vb = problem.vertex[b];

    //Coordinates
    double xa = va(0), ya = va(1);
    double xb = vb(0), yb = vb(1);

    //integer distances
    int dy = std::abs(ya - yb);
    int dx = std::abs(xa - xb);

    //Base turn count
    int n_turns = 1;
    bool fourth_quadrant = (xa < xb) && (ya < yb);
    bool second_quadrant = (xa < xb) && (ya > yb);
    bool first_quadrant = (xa > xb) && (ya > yb);
    bool third_quadrant = (xa > xb) && (ya < yb);

    //Degen case
    if (a % n_cols == 0 && first_quadrant)
        return {5 + std::abs(1 - dy), 5};
    else if ( a / n_cols == 0 && dy == 0 && fourth_quadrant)
        return {5 + std::abs(1 - dx), 5};

    if (!third_quadrant) n_turns += 2;

    //Base length count
    int length = n_turns + dx + dy;

    //Geometric corrections
    if ((first_quadrant && dx == 0) || (fourth_quadrant && dy==0)) length += 1;
    else if (fourth_quadrant || first_quadrant) length -= 1;

    return {length, n_turns};
}

cost_t
path_engine::wn_dist(int a, int b) const
{
    const auto& va = problem.vertex[a];
    const auto& vb = problem.vertex[b];

    //Coordinates
    double xa = va(0), ya = va(1);
    double xb = vb(0), yb = vb(1);

    //integer distances
    int dy = std::abs(ya - yb);
    int dx = std::abs(xa - xb);

    //Base turn count
    int n_turns = 1;
    bool second_quadrant = (xa < xb) && (ya > yb);
    bool first_quadrant = (xa > xb) && (ya > yb);
    bool third_quadrant = (xa > xb) && (ya < yb);

    //Degen case
    if (a % problem.seabed.cols() == 0 && (dx == 0) && third_quadrant)
        return {5 + std::abs(1 - dy), 5};
    else if (a / problem.seabed.cols() == problem.seabed.rows() && (dy == 0) && second_quadrant)
        return {5 + std::abs(1 - dx), 5};

    if (!first_quadrant) n_turns += 2;

    //Base length count
    int length = n_turns + dx + dy;

    //Geometric corrections
    //if ((third_quadrant && dx == 0) || (second_quadrant && (dx + dy==0)) length += 1;
    if ((third_quadrant && dx == 0) || (second_quadrant && dy==0)) length += 1;
    else if (second_quadrant || third_quadrant) length -= 1;

    return {length, n_turns};
}

cost_t
path_engine::ss_dist(int a, int b) const
{
    const auto& va = problem.vertex[a];
    const auto& vb = problem.vertex[b];

    //Integer coordinates
    int xa = int(va(0)), ya = int(va(1));
    int xb = int(vb(0)), yb = int(vb(1));

    //Integer distances (valid)
    int dy = std::abs(ya - yb);
    int dx = std::abs(xa - xb);

    //bool fourth_quadrant = (xa < xb) && (ya < yb);
    bool second_quadrant = (xa < xb) && (ya > yb);
    bool first_quadrant = (xa > xb) && (ya > yb);
    //bool third_quadrant = (xa > xb) && (ya < yb);

    //Base turn count
    bool backward = (yb <= ya);     // need to reverse direction
    bool straight = !backward & (xb == xa);

    //Edge case
    if (backward && ((first_quadrant || second_quadrant) || dx == 0) && dx <= 1)
        return { 4 + dx + dy, 4};
    else if (a == b)
        return {4,4};
    else if (dy==0)
        return {4 + std::abs(dx - 2),4};

    return s_shaped_dist(dx, dy, backward, straight);
}

cost_t
path_engine::sn_dist(int a, int b) const
{
    //Degen case
    if (a == b && (a - horizontal_bound) % n_rows == (n_rows - 1))
        return {7, 6};
    else if (a == b)
        return {5, 4};

    const auto& va = problem.vertex[a];
    const auto& vb = problem.vertex[b];

    //Integer coordinates
    int xa = int(va(0)), ya = int(va(1));
    int xb = int(vb(0)), yb = int(vb(1));

    //Integer distances (valid)
    int dy = std::abs(ya - yb);
    int dx = std::abs(xa - xb);

    //Base turn count
    int n_turns = 2;

    //Base length count;
    int length = n_turns + dx + dy;

    //Geometric corrections
    if (dx == 0) {
        n_turns += 2;
        length += 1;
    } else
        length -= 1;
    return {length, n_turns};
}

cost_t
path_engine::se_dist(int a, int b) const{
    return wn_dist(b,a);
}

cost_t
path_engine::sw_dist(int a, int b) const{
    return en_dist(b,a);
}

cost_t
path_engine::nn_dist(int a, int b) const
{
    return ss_dist(b,a);
}

cost_t
path_engine::ns_dist(int a, int b) const
{
    if ((a - horizontal_bound) % n_rows == 0 && a == b)
        return {7,6};
    else if (a == b)
        return {5,4};
    return sn_dist(b,a);
}

cost_t
path_engine::ne_dist(int a, int b) const
{
    return ws_dist(b,a);
}

cost_t
path_engine::nw_dist(int a, int b) const
{
    return es_dist(b,a);
}

cost_t
path_engine::s_shaped_dist(int dx, int dy, bool backward, bool straight) {
    cost_t c = {dx + dy, 0};
    c.turns += 2*(1 - straight);
    c.turns += 2*(backward);
    c.length += 2*(backward);
    return c;
}