#include "../../include/engines/path_engine.hpp"

cost_t path_engine::dist(const segment &a, const segment &b) const {
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
    direction dir = seg.source < horizontal_bound ? direction::E : direction::S;
    if (is_node(seg.target))
        dir = seg.target < seg.source ? flip_direction(dir) : dir;
    else
        dir = seg.target == REVERSED_NULL_NODE ? flip_direction(dir) : dir;
    return dir;
}

segment path_engine::flip_segment(const segment &s){
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