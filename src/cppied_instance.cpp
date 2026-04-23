#include "../include/cppied_instance.hpp"

void cppied_instance_base::transform_log1p() {
    assert(pod.maxCoeff() < 1);
    assert(pod.minCoeff() >= 0);
    pod.array() = (-pod.array()).log1p().abs();
    req.array() = (-req.array()).log1p().abs();
}

cppied_solution cppied_instance::initialize_solution() {
    return {
            {},
        Eigen::VectorXd::Zero(req.size()),
        cost_t{0,0}
    };
}

void cppied_instance_base::initialize_cells() {
    cells.resize(2, seabed.rows()*seabed.cols());
    for (int i=0; i< seabed.rows(); i++){
        for (int j=0; j< seabed.cols(); j++) {
            int idx = (i * int(seabed.cols())) + j;
            float x = float(j) + 0.5f;
            float y = float(i) + 0.5f;
            cells.col(idx) << x, y;
        }
    }
}

void cppied_instance_base::initialize_sparse_adj() {
    std::vector<Eigen::Triplet<int>> adjacency_list;
    const int V = static_cast<int>(((seabed.rows() + 1) * seabed.cols())
                  + (seabed.rows() * (seabed.cols() + 1)));
    adjacency_list.reserve(6 * V);
    for (int v=0; v<vertex.size(); v++){
        std::vector<int> adj_vec;
        bool horizontal = v < (seabed.rows()+1)*seabed.cols();
        set_adjacency_vec(v, adj_vec, horizontal);
        for(int u : adj_vec)
            adjacency_list.emplace_back(v,u, 1);
//        for (int v2=v1+1; v2<vertex.size(); v2++){
//            if (Point2F::L1(vertex[v1], vertex[v2]) <= 1.0 + 1e-5
//                && adjacent(vertex[v1], vertex[v2])) {
//                adjacency_list.emplace_back(v1, v2, 1);
//                adjacency_list.emplace_back(v2, v1, 1);
//            }
//        }
    }
    adj.resize(
            int(vertex.size()),
            int(vertex.size())
    );
    adj.reserve(int(adjacency_list.size()));
    adj.setFromTriplets(adjacency_list.begin(), adjacency_list.end());
    adj.makeCompressed();
}

void cppied_instance_base::initialize_vertices() {
    const int V = static_cast<int>(((seabed.rows() + 1) * seabed.cols())
            + (seabed.rows() * (seabed.cols() + 1)));
    vertex.reserve(V);
    for (int i =0; i<=seabed.rows(); i++)
        for (int j = 0; j < seabed.cols(); j++)
            vertex.emplace_back(j + 0.5, i);

    for (int j=0; j<=seabed.cols(); j++)
        for (int i = 0; i < seabed.rows(); i++)
            vertex.emplace_back(float(j), i + 0.5);

}

void cppied_instance_base::initialize_sparse_pod() {
    std::vector<Eigen::Triplet<double>> set_cover_list;
    const int V = static_cast<int>(((seabed.rows() + 1) * seabed.cols())
                  + (seabed.rows() * (seabed.cols() + 1)));
    set_cover_list.reserve(V * 2* max_range);

    for (int i =0; i<=seabed.rows(); i++) {
        const int offset = static_cast<int>(i * seabed.cols());
        for (int j = 0; j < seabed.cols(); j++) {
            int idx = offset + j;
            std::vector<int> C;
            set_cover(idx, C, true);
            for (int c : C) {
                const int dy = static_cast<int>(std::floor(
                        std::abs(vertex[idx].y - cells(1,c))));
                int c_type = seabed(
                        int(floor(cells(1, c))),
                        int(floor(cells(0, c))));
                set_cover_list.emplace_back(c, idx,pod(c_type,dy));
            }
        }
    }
    //std::cout << "(vertical) " << std::flush;
    for (int j=0; j<=seabed.cols(); j++) {
        const int offset = static_cast<int>((seabed.rows() + 1) * seabed.cols() + (j*seabed.rows()));
        for (int i=0; i<seabed.rows(); i++){
            int idx = offset + i;
            std::vector<int> C;
            set_cover(idx, C, false);
            for (int c : C){
                const int dx = static_cast<int>(std::floor(
                        std::abs(vertex[idx].x - cells(0,c))));
                int c_type = seabed(
                        int(floor(cells(1, c))),
                        int(floor(cells(0, c))));
                set_cover_list.emplace_back(c, idx,pod(c_type,dx));
            }
        }
    }
    //std::cout << "(copressing)... " << std::flush;
    s_pod.resize(
            cells.cols(),        // number of rows
            int(vertex.size())     // number of columns
    );
    s_pod.setFromTriplets(set_cover_list.begin(), set_cover_list.end());
    s_pod.makeCompressed();
}

void cppied_instance_base::initialize_intermediate() {
    //std::cout << "\nInitializing .. " << std::flush;
    transform_log1p();
    max_range = int(pod.cols());
    initial_position = static_cast<int>(seabed.cols() * max_range);

    //std::cout << "cells coordinates... " << std::flush;
    initialize_cells();
    //std::cout << "vertices coordinates... " << std::flush;
    initialize_vertices();
    //std::cout << "sparse adj coordinates... " << std::flush;
    initialize_sparse_adj();
    //std::cout << "sparse pod coordinates... " << std::flush;
    initialize_sparse_pod();
    //std::cout << "done." << std::endl;
}

bool cppied_instance_base::adjacent(const Point2F &a, const Point2F &b) {
    Point2F vaf = {std::floor(a.x), std::floor(a.y)};
    Point2F vac = {std::ceil(a.x), std::ceil(a.y)};

    Point2F vbf = {std::floor(b.x), std::floor(b.y)};
    Point2F vbc = {std::ceil(b.x), std::ceil(b.y)};

    bool is_adjacent = vaf == vbf;
    is_adjacent |= vaf == vbc;
    is_adjacent |= vac == vbf;
    is_adjacent |= vac == vbc;

    return is_adjacent;
}

void cppied_instance_base::validate_parameters() {
    if (seabed.minCoeff() < 0)
        throw algo_exception("Seabed contains negative habitat indices.");
    if (seabed.maxCoeff() >= pod.rows())
        throw algo_exception("Pod matrix does not contain enough rows for the seabed entries.");
}

void cppied_instance_base::set_cover(int v, std::vector<int> &C, bool horizontal) {
    const Point2F v_p = vertex[v];
    int cell1, cellm, step;
    if (horizontal){
        const int row1 = std::max(0,static_cast<int>(v_p.y) - max_range);
        const int rowm = std::min(static_cast<int>(seabed.rows())-1, static_cast<int>(v_p.y) + max_range - 1);
        const int col = static_cast<int>(std::floor(v_p.x));
        cell1 = (row1 * static_cast<int>(seabed.cols())) + col;
        cellm = (rowm * static_cast<int>(seabed.cols())) + col;
        step = static_cast<int>(seabed.cols());
    } else {
        const int col1 = std::max(0, static_cast<int>(v_p.x) - max_range);
        const int colm = std::min(static_cast<int>(seabed.cols())-1, static_cast<int>(v_p.x) + max_range - 1);
        const int row = static_cast<int>(std::floor(v_p.y));
        cell1 = (row * static_cast<int>(seabed.rows())) + col1;
        cellm = (row * static_cast<int>(seabed.rows())) + colm;
        step = 1;
    }
    C.reserve(((cellm - cell1) / step) + 1);
    for(int c = cell1; c<= cellm; c+=step)
        C.push_back(c);
}

void cppied_instance_base::set_adjacency_vec(int v, std::vector<int> &adj_vec, bool horizontal) {
    int offset;
    int step;
    if (horizontal) {
        offset = static_cast<int>(((seabed.rows() + 1) * seabed.cols())
                                  + (seabed.rows() * std::floor(vertex[v].x))
                                  + (vertex[v].y - 1));
        step   = static_cast<int>(seabed.rows());
    } else {
        offset = static_cast<int>(
                (seabed.cols()*std::floor(vertex[v].y)) + (vertex[v].x -1)
                );
        step = static_cast<int>(seabed.cols());
    }
    adj_vec = {v-1, v+1, offset, offset + 1, offset + step, offset + step + 1};
    auto filter = [&](int u){
        if ((u <0) || (u>=vertex.size()))
            return true;
        return Point2F::L1(vertex[v], vertex[u]) > 1.0 + 1e-5 || !adjacent(vertex[v], vertex[u]);
    };
    std::erase_if(adj_vec, filter);
}
