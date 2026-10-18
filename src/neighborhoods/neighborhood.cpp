#include "../../include/neighborhoods/neighborhood.hpp"

segment neighborhood::trim(cppied_solution &pSol, neighborhood::sVecIt seg) {
    return trim(pSol, *seg);
}

segment neighborhood::trim(cppied_solution &pSol, segment seg) {
    segment new_seg = seg;
    if (geometry.is_horizontal(new_seg))
        geometry.correct_segment_direction(new_seg, direction::E);
    else
        geometry.correct_segment_direction(new_seg, direction::S);
    if (path_engine::is_node(new_seg.target)){
        int u = new_seg.source;
        for (; u<= new_seg.target; ++u){
            if (!coverage.can_remove(pSol, {u, path_engine::NULL_NODE}))
                break;
        }
        int v = new_seg.target;
        for (; v>=u; --v){
            if (!coverage.can_remove(pSol, {v, path_engine::NULL_NODE}))
                break;
        }
        if (v < u)
            new_seg = {path_engine::NULL_NODE, path_engine::NULL_NODE};
        else if (v==u)
            new_seg = {u, path_engine::NULL_NODE};
        else
            new_seg = {u,v};
        if (path_engine::is_node(new_seg.source))
            geometry.correct_segment_direction(new_seg, geometry.get_direction(seg));
    } else if (coverage.can_remove(pSol, new_seg))
        new_seg = {path_engine::NULL_NODE, path_engine::NULL_NODE};
    return new_seg;
}

std::vector<segment> neighborhood::getCompleteCoverageSegments(const iRectangle &box) {
    std::vector<segment> segments = getCompleteCoverageHorizontalSegments(box);
    std::vector<segment> vertical_segments = getCompleteCoverageVerticalSegments(box);
    segments.insert(segments.end(), vertical_segments.begin(), vertical_segments.end());
    return segments;
}

std::vector<segment> neighborhood::getCompleteCoverageHorizontalSegments(const iRectangle &box) {
    std::vector<segment> segment_set;
    int row = std::max(0, box.lr.y - problem.max_range);
    int bound = std::min(geometry.n_rows, box.ul.y + problem.max_range);

    //Horizontal segments
    for (; row <= bound; ++row){
        segment s{row * geometry.n_cols + box.ul.x,row * geometry.n_cols + box.lr.x - 1};
        if (s.source == s.target) s.target = path_engine::NULL_NODE;
        segment_set.push_back(s);
    }
    return segment_set;
}

std::vector<segment> neighborhood::getCompleteCoverageVerticalSegments(const iRectangle &box) {
    std::vector<segment> segment_set;
    //Vertical segments
    int col = std::max(0, box.lr.x - problem.max_range);
    int bound = std::min(geometry.n_cols, box.ul.x + problem.max_range);

    for (; col <= bound; ++col){
        segment s{geometry.horizontal_bound + (col * geometry.n_rows) + box.ul.y,
                    geometry.horizontal_bound + col * geometry.n_rows + box.lr.y - 1};
        if (s.source == s.target) s.target = path_engine::NULL_NODE;
        segment_set.push_back(s);
    }
    return segment_set;
}

