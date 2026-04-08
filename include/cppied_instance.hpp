#pragma once

#include "structures.hpp"
#include "../include/utils.hpp"

class cppied_instance{
public:
    MatrixXdRow<int> seabed;
    MatrixXdRow<double> pod;
    Eigen::VectorXd req;
    Eigen::SparseMatrix<int> adj;
    Eigen::SparseMatrix<double> s_pod;
    std::vector<Eigen::Vector2f> vertex;
    Eigen::Matrix2Xf cells;
    int initial_position;
    int max_range;

    cppied_instance(MatrixXdRow<int> pSeabed,
                    MatrixXdRow<double> pPod,
                    MatrixXdRow<double> pReq){
        seabed = std::move(pSeabed);
        pod = std::move(pPod);
        req = Eigen::Map<Eigen::VectorXd>(pReq.data(), pReq.size());

        initialize_intermediate();
        validate_parameters();
    }
    cppied_solution initialize_solution();
protected:
    void transform_log1p();
    void initialize_intermediate();
    bool adjacent (const Eigen::Vector2f& a,
                   const Eigen::Vector2f& b);
    void validate_parameters();
};
