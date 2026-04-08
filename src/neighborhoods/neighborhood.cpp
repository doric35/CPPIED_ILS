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
    if (is_node(new_seg.target)){
        int u = new_seg.source;
        for (; u<= new_seg.target; ++u){
            if (!coverage.can_remove(pSol, {u, NULL_NODE}))
                break;
        }
        int v = new_seg.target;
        for (; v>=u; --v){
            if (!coverage.can_remove(pSol, {v, NULL_NODE}))
                break;
        }
        if (v < u)
            new_seg = {NULL_NODE, NULL_NODE};
        else if (v==u)
            new_seg = {u, NULL_NODE};
        else
            new_seg = {u,v};
        if (is_node(new_seg.source))
            geometry.correct_segment_direction(new_seg, geometry.get_direction(seg));
    } else if (coverage.can_remove(pSol, new_seg))
        new_seg = {NULL_NODE, NULL_NODE};
    return new_seg;
}