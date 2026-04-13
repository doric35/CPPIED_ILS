#include "../include/cppied_instance.hpp"

void cppied_instance::transform_log1p() {
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

void cppied_instance::initialize_cells() {
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

void cppied_instance::initialize_sparse_adj() {
    std::vector<Eigen::Triplet<int>> adjacency_list;
    const int V = ((seabed.rows() + 1) * seabed.cols())
                  + (seabed.rows() * (seabed.cols() + 1));
    adjacency_list.reserve(6 * V);
    for (int v1=0; v1<vertex.size(); v1++){
        for (int v2=v1+1; v2<vertex.size(); v2++){
            if ((vertex[v1].array() - vertex[v2].array()).abs().sum() <= 1.0
                && adjacent(vertex[v1], vertex[v2])) {
                adjacency_list.emplace_back(v1, v2, 1);
                adjacency_list.emplace_back(v2, v1, 1);
            }
        }
    }
    adj.resize(
            int(vertex.size()),
            int(vertex.size())
    );
    adj.reserve(int(adjacency_list.size()));
    adj.setFromTriplets(adjacency_list.begin(), adjacency_list.end());
    adj.makeCompressed();
}

void cppied_instance::initialize_vertices() {
    const int V = ((seabed.rows() + 1) * seabed.cols())
            + (seabed.rows() * (seabed.cols() + 1));
    vertex.reserve(V);
    for (int i =0; i<=seabed.rows(); i++)
        for (int j = 0; j < seabed.cols(); j++)
            vertex.emplace_back(j + 0.5, i);

    for (int j=0; j<=seabed.cols(); j++)
        for (int i = 0; i < seabed.rows(); i++)
            vertex.emplace_back(float(j), i + 0.5);

}

void cppied_instance::initialize_sparse_pod() {
    std::vector<Eigen::Triplet<double>> set_cover_list;
    const int V = ((seabed.rows() + 1) * seabed.cols())
                  + (seabed.rows() * (seabed.cols() + 1));
    set_cover_list.reserve(V * 2* max_range);

    for (int i =0; i<=seabed.rows(); i++) {
        const int offset = i * seabed.cols();
        for (int j = 0; j < seabed.cols(); j++) {
            Eigen::ArrayXf dx = cells.row(0).array() - vertex[offset + j](0);          // row(0) = x coordinates
            Eigen::ArrayXf dy = cells.row(1).array() - vertex[offset + j](1);
            Eigen::Array<bool, Eigen::Dynamic, 1> mask = (dx.abs() < 1e-6f) &&
                                                         (dy.abs() < float(pod.cols()));
            int idx = offset + j;
            for (int c = 0; c < mask.size(); c++)
                if (mask(c))
                    set_cover_list.emplace_back(c,idx,
                                                pod(seabed(
                                                            int(floor(cells(1, c))),
                                                            int(floor(cells(0, c)))),
                                                    int(floor(abs(dy(c))))));
        }
    }
    //std::cout << "(vertical) " << std::flush;
    for (int j=0; j<=seabed.cols(); j++) {
        const int offset = (seabed.rows() + 1) * seabed.cols() + (j*seabed.rows());
        for (int i=0; i<seabed.rows(); i++){
            Eigen::ArrayXf dx = cells.row(0).array() - vertex[offset + i](0);          // row(0) = x coordinates
            Eigen::ArrayXf dy = cells.row(1).array() - vertex[offset + i](1);
            Eigen::Array<bool, Eigen::Dynamic, 1> mask = (dy.abs() < 1e-6f) &&
                                                         (dx.abs() < float(pod.cols()));
            int idx = offset + i;
            for (int c = 0; c < mask.size(); c++)
                if (mask(c)) set_cover_list.emplace_back(c, idx,
                                                         pod(seabed(
                                                                     int(floor(cells(1,c))),
                                                                     int(floor(cells(0,c)))),
                                                             int(floor(abs(dx(c))))));
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

void cppied_instance::initialize_intermediate() {
    std::cout << "\nInitializing .. " << std::flush;
    transform_log1p();
    max_range = int(pod.cols());
    initial_position = seabed.cols() * max_range;

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

bool cppied_instance::adjacent(const Eigen::Vector2f &a, const Eigen::Vector2f &b) {
    Eigen::Vector2f vaf = a.array().floor();
    Eigen::Vector2f vac = a.array().ceil();

    Eigen::Vector2f vbf = b.array().floor();
    Eigen::Vector2f vbc = b.array().ceil();

    bool is_adjacent = vaf == vbf;
    is_adjacent |= vaf == vbc;
    is_adjacent |= vac == vbf;
    is_adjacent |= vac == vbc;

    return is_adjacent;
}

void cppied_instance::validate_parameters() {
    if (seabed.minCoeff() < 0)
        throw algo_exception("Seabed contains negative habitat indices.");
    if (seabed.maxCoeff() >= pod.rows())
        throw algo_exception("Pod matrix does not contain enough rows for the seabed entries.");
}

