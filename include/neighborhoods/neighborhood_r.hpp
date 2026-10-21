#pragma once

#include <utility>

#include "neighborhood.hpp"
#include "gurobi_c++.h"
#include "../algorithms/MaximumReplacement.hpp"
#include "../algorithms/MinimumPrice.hpp"

struct ReallocationSet{
    std::vector<ReplacementMove> moves;
    std::vector<GRBVar> variables;
};

struct PricingSolution{
    segment s;
    double value;
    int R;
    cost_t costs;
};

class TimeOutError : public std::runtime_error{
    using std::runtime_error::runtime_error;
};

class neighborhood_r : public neighborhood {
public:
    template<typename... Args>
    explicit neighborhood_r(Args&&... args) : neighborhood(std::forward<Args>(args)...),
        coverageConstraints(geometry.n_rows*geometry.n_cols),
        vertexCoveragePrices(geometry.n_vertex),
        coverageDuals(geometry.n_rows*geometry.n_cols),
        lengthDual(0.0)
        {
        vertexCoveragePrices.setZero();
        coverageDuals.setZero();
    }

    bool local_search() override;

protected:
    GRBEnv *env = nullptr;
    GRBModel *model = nullptr;
    std::vector<GRBConstr> coverageConstraints;
    std::vector<GRBConstr> reallocationConstraints{};
    GRBConstr pathLengthConstraint{};

    GRBLinExpr *Z1 = nullptr;
    GRBLinExpr *Z2 = nullptr;

    std::vector<int> reallocationSelection{};
    std::vector<int>             unsatisfiedCells{};
    std::vector<ReallocationSet> reallocationSets{};

    Eigen::VectorXd vertexCoveragePrices;
    Eigen::VectorXd coverageDuals;
    Eigen::VectorXd reallocationDuals;
    double lengthDual;

    void reset();
    cost_t removalGain(int index);
    cost_t replacementGain(int index, const segment& s);
    void setSelectReallocations();
    std::vector<cost_t> getRemovalGains();
    void setInitialReallocationSets();
    void pushInitialSegmentReallocation(int index);

    void setRemovalCoverage();

    void constructGurobiModel();
    void initializeGurobiEnvironment();
    void initializeGurobiModel();
    void constructLinearProgram();
    void constructInitialVariables();
    void constructReallocationSetVariables(int index);
    void constructObjectives();
    void constructLengthObjective();
    void constructTurnsObjective();
    void constructConstraints();
    void constructCoverageConstraints();
    void constructReallocationConstraints();
    void constructLengthConstraints();

    void columnGeneration();
    void runColumnGenerationIteration();
    void runColumnGenerationFirstStage();
    void runColumnGenerationSecondStage();
    void generateColumn(PricingSolution& pricingSolution);
    [[nodiscard]] bool columnGenerationHasConverged() const;

    template<typename PricingPhaseSolver>
    PricingSolution solvePricingSubproblem();
    void setDualVariables();
    void setCoverageDuals();
    void setReallocationDuals();
    void setLengthDual();
    void setVertexPrices();

    void setLexicographicBinaryProgram();
    void setReallocationModelWarmStart();

    void retrieveSolution();
    void safeSolutionRetrieve();
    segment reallocation(int index);

    void throwIfLpTimedOut();
private:
    bool columnGenerationConvergedFlag = false;
};