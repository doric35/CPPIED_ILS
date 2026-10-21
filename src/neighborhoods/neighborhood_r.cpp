#include "../../include/neighborhoods/neighborhood_r.hpp"

bool neighborhood_r::local_search() {
    cost_t initial_cost = solution.cost;
    reset();
    setSelectReallocations();
    constructGurobiModel();
    reallocationDuals.resize(static_cast<int>(reallocationSets.size()));
    reallocationDuals.setZero();

    try {
        columnGeneration();
    } catch (TimeOutError& E){
        coverage.reset(solution);
        return false;
    }
    setLexicographicBinaryProgram();
    setReallocationModelWarmStart();

    model->set(GRB_DoubleParam_TimeLimit, remaining_time());
    model->update();
    model->optimize();
    retrieveSolution();
    return solution.cost < initial_cost;
}

void neighborhood_r::reset() {
    if (Z1) {
        Z1->clear();
        delete Z1;
        Z1 = nullptr;
    }
    if (Z2) {
        Z2->clear();
        delete Z2;
        Z2 = nullptr;
    }
    reallocationSelection.clear();
    reallocationSets.clear();
    coverageDuals.setZero();
    reallocationDuals.setZero();
    lengthDual = 0.0;
    columnGenerationConvergedFlag = false;
    delete model;
    model = nullptr;
    delete env;
    env = nullptr;
}

void neighborhood_r::constructGurobiModel() {
    initializeGurobiEnvironment();
    initializeGurobiModel();
    setInitialReallocationSets();
    constructLinearProgram();
    model->set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
    model->update();
}

void neighborhood_r::setSelectReallocations() {
    std::vector<cost_t> gains = getRemovalGains();
    const int index_bound = static_cast<int>(solution.path.size());
    auto pairs_view = std::views::iota(1, index_bound)
                      | std::views::filter([](int i){ return i% 2 == 0; });
    auto impairs_view = std::views::iota(1, index_bound)
                      | std::views::filter([](int i){ return i% 2 == 1; });
    auto sum_gain = [&](cost_t cost, int i){ return cost + gains[i]; };
    cost_t pair_gain = std::ranges::fold_left(pairs_view, cost_t{0,0}, sum_gain);
    cost_t impairs_gain = std::ranges::fold_left(impairs_view, cost_t{0,0}, sum_gain);
    std::vector<cost_t> cumulative_gains{pair_gain, impairs_gain};
    int s = ris_heuristic::softmaxSample(cumulative_gains, rng);

    if (s)
        reallocationSelection = std::ranges::to<std::vector<int>>(impairs_view);
    else
        reallocationSelection = std::ranges::to<std::vector<int>>(pairs_view);
}

std::vector<cost_t> neighborhood_r::getRemovalGains() {
    std::vector<cost_t> gains(solution.path.size(), cost_t{0,0});
    for(int i=1; i< solution.path.size(); ++i)
        gains[i] = removalGain(i);
    return gains;
}

void neighborhood_r::initializeGurobiEnvironment() {
    env = new GRBEnv(true);
    auto log_item = ctx.config.find("WORKING_DIRECTORY");
    if (log_item != ctx.config.end()){
        std::filesystem::path log_path = log_item->second;
        log_path = log_path / "reallocation.log";
        env->set("LogFile", log_path);
    } else
        env->set("LogFile", "reallocation.log");
    env->set(GRB_IntParam_OutputFlag, 0);
    env->set(GRB_IntParam_LogToConsole, 0);
    env->set(GRB_IntParam_ThreadLimit, 1);
    env->start();
}

void neighborhood_r::initializeGurobiModel() {
    model = new GRBModel(*env);
}

void neighborhood_r::setInitialReallocationSets() {
    reallocationSets.resize(solution.path.size(),{});
    for (int r : reallocationSelection)
        pushInitialSegmentReallocation(r);
}

void neighborhood_r::pushInitialSegmentReallocation(int index) {
    ReplacementMove self_move{solution.path[index], replacementGain(index, solution.path[index])};
    reallocationSets[index].moves.push_back(self_move);
    MaximumReplacement replacement_algorithm(problem, coverage, geometry, solution, index);
    ReplacementMove move = replacement_algorithm.solve();
    if (path_engine::is_node(move.s.source)) {
        move.gain = replacementGain(index, move.s);
        reallocationSets[index].moves.push_back(move);
    }
}

void neighborhood_r::constructLinearProgram() {
    constructInitialVariables();
    constructObjectives();
    constructConstraints();
}

void neighborhood_r::constructInitialVariables() {
    for (int i : reallocationSelection)
        constructReallocationSetVariables(i);
}

void neighborhood_r::constructReallocationSetVariables(int index) {
    for (auto& move : reallocationSets[index].moves)
        reallocationSets[index].variables.push_back(model->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, nullptr));
}

void neighborhood_r::constructObjectives() {
    constructLengthObjective();
    constructTurnsObjective();
}

void neighborhood_r::constructLengthObjective() {
    Z1 = new GRBLinExpr(0.0);
    for (auto& set : reallocationSets){
        for (std::size_t i = 0; i< set.moves.size(); i++)
            *Z1 += set.moves[i].gain.length * set.variables[i];
    }
}

void neighborhood_r::constructTurnsObjective() {
    Z2 = new GRBLinExpr(0.0);
    for (auto& set : reallocationSets){
        for (std::size_t i = 0; i< set.moves.size(); i++)
            *Z2 += set.moves[i].gain.turns * set.variables[i];
    }
}

void neighborhood_r::constructConstraints() {
    constructCoverageConstraints();
    constructReallocationConstraints();
    constructLengthConstraints();
}

void neighborhood_r::constructCoverageConstraints() {
    setRemovalCoverage();
    coverage.unsatisfied_cells(solution, unsatisfiedCells);
    std::vector<GRBLinExpr> rhs(geometry.n_rows*geometry.n_cols, 0.0);
    for (int i : reallocationSelection){
        for (int k =0; k<reallocationSets[i].moves.size(); k++){
            segment non_oriented = geometry.getNonOriented(reallocationSets[i].moves[k].s);
            int u = non_oriented.source;
            int v = non_oriented.target;
            if (!path_engine::is_node(v)) v = non_oriented.source;
            for (int n = u; n <= v; n++){
                for (Eigen::SparseMatrix<double>::InnerIterator c(problem.s_pod, n); c; ++c)
                    rhs[c.index()] += reallocationSets[i].variables[k] * c.value();
            }
        }
    }
    for (int c : unsatisfiedCells)
        coverageConstraints[c] = model->addConstr(rhs[c] >= problem.req[c] - solution.coverage[c]);
}

void neighborhood_r::setRemovalCoverage() {
    for (int i : reallocationSelection)
        coverage.remove(solution, solution.path[i]);
}

void neighborhood_r::constructReallocationConstraints() {
    reallocationConstraints.resize(solution.path.size());
    for (int i : reallocationSelection){
        GRBLinExpr select_unit = 0.0;
        for (auto& var : reallocationSets[i].variables)
            select_unit += var;
        reallocationConstraints[i] = model->addConstr(select_unit <= 1);
    }
}

void neighborhood_r::constructLengthConstraints() {
    pathLengthConstraint = model->addConstr(*Z1 <= solution.cost.length);
}

void neighborhood_r::columnGeneration() {
    while (!columnGenerationHasConverged())
        runColumnGenerationIteration();
}

void neighborhood_r::runColumnGenerationIteration() {
    runColumnGenerationFirstStage();
    runColumnGenerationSecondStage();
}

void neighborhood_r::runColumnGenerationFirstStage() {
    model->setObjective(*Z1, GRB_MINIMIZE);
    model->set(GRB_DoubleParam_TimeLimit, remaining_time());
    model->update();
    model->optimize();
    throwIfLpTimedOut();
    PricingSolution reallocation = solvePricingSubproblem<FirstStageMinimumPrice>();
    if (reallocation.value < -1e-6)
        generateColumn(reallocation);
    else
        columnGenerationConvergedFlag = true;
}

void neighborhood_r::runColumnGenerationSecondStage() {
    double optimal_length = model->get(GRB_DoubleAttr_ObjVal);
    pathLengthConstraint.set(GRB_DoubleAttr_RHS, optimal_length);
    model->setObjective(*Z2, GRB_MINIMIZE);
    model->set(GRB_DoubleParam_TimeLimit, remaining_time());
    model->update();
    model->optimize();
    throwIfLpTimedOut();
    PricingSolution reallocation = solvePricingSubproblem<SecondStageMinimumPrice>();
    if (reallocation.value < -1e-6) {
        generateColumn(reallocation);
        columnGenerationConvergedFlag = false;
    }
}

bool neighborhood_r::columnGenerationHasConverged() const {
    return columnGenerationConvergedFlag;
}

void neighborhood_r::setLexicographicBinaryProgram() {
    model->setObjective(GRBLinExpr(0.0));
    model->setObjectiveN(*Z1, 0, 2);
    model->setObjectiveN(*Z2, 1, 1);

    for (auto& set : reallocationSets){
        for (auto& var : set.variables)
            var.set(GRB_CharAttr_VType, GRB_BINARY);
    }
    model->remove(pathLengthConstraint);
}

void neighborhood_r::setReallocationModelWarmStart() {
    for (auto& set : reallocationSets){
        if (!set.variables.empty()) {
            set.variables[0].set(GRB_DoubleAttr_Start, 1.0);
            for (int i = 1; i < set.variables.size(); ++i)
                set.variables[i].set(GRB_DoubleAttr_Start, 0.0);
        }
    }
}

void neighborhood_r::retrieveSolution() {
    int status = model->get(GRB_IntAttr_Status);
    if (status == GRB_OPTIMAL || status == GRB_SUBOPTIMAL) {
        // A feasible solution exists, safe to retrieve
        safeSolutionRetrieve();
        reset();
    } else if (status == GRB_INF_OR_UNBD || status == GRB_INFEASIBLE) {
        // No feasible solution found
        reset();
        std::cerr << "[Error] Model infeasible or unbounded\n";
        throw std::runtime_error("Infeasibility or unboundedness found in reallocation model.\n");
    } else {
        // Possibly interrupted due to TimeLimit
        reset();
        std::cerr << "[Warning] Optimization interrupted in reallocation model, using best solution so far\n";
        coverage.reset(solution);
    }
}

void neighborhood_r::throwIfLpTimedOut() {
    int status = model->get(GRB_IntAttr_Status);
    if (status != GRB_OPTIMAL)
        throw TimeOutError("[WARNING] : Reallocation LP encountered a timeout.\n");
}

void neighborhood_r::safeSolutionRetrieve() {
    int j =0;
    for (int i=1; i< solution.path.size(); ++i){
        if (j < reallocationSelection.size() && reallocationSelection[j] == i) {
            solution.path[i] = reallocation(i);
            ++j;
        }
    }
    auto nullSegment = [&](const segment& s){ return !path_engine::is_node(s.source); };
    std::erase_if(solution.path, nullSegment);
    coverage.reset(solution);
    solution.cost = geometry.cost(solution);
}

segment neighborhood_r::reallocation(int index) {
    segment replacement = {path_engine::NULL_NODE, path_engine::NULL_NODE};
    const auto& set = reallocationSets[index];
    for (int i=0; i< set.moves.size(); ++i)
    {
        double move_val = set.variables[i].get(GRB_DoubleAttr_X);
        if (move_val > 0.5) replacement = set.moves[i].s;
    }
    return replacement;
}

template<typename PricingPhaseSolver>
PricingSolution neighborhood_r::solvePricingSubproblem() {
    setDualVariables();
    setVertexPrices();
    Reallocation argmin{{path_engine::NULL_NODE, path_engine::NULL_NODE}, MAX_INT, -1, {0, 0}};
    for (int r : reallocationSelection){
        PricingPhaseSolver solver(vertexCoveragePrices, reallocationDuals(r), lengthDual,
                                  problem, coverage, geometry, solution, r);
        Reallocation reallocation = solver.solve();
        if (reallocation.value < argmin.value) argmin = reallocation;
    }
    cost_t costs{0, 0};
    if (path_engine::is_node(argmin.S.source))
        costs = replacementGain(argmin.R, argmin.S);
    PricingSolution sol{argmin.S, argmin.value, argmin.R, costs};
    return sol;
}

void neighborhood_r::setDualVariables() {
    setCoverageDuals();
    setReallocationDuals();
    setLengthDual();
}

void neighborhood_r::setCoverageDuals() {
    for (int i : unsatisfiedCells)
        coverageDuals(i) = coverageConstraints[i].get(GRB_DoubleAttr_Pi);
}

void neighborhood_r::setReallocationDuals() {
    for (int r : reallocationSelection)
        reallocationDuals(r) = reallocationConstraints[r].get(GRB_DoubleAttr_Pi);
}

void neighborhood_r::setLengthDual() {
    lengthDual = pathLengthConstraint.get(GRB_DoubleAttr_Pi);
}

void neighborhood_r::setVertexPrices() {
    for (int u=0; u<geometry.n_vertex; ++u){
        vertexCoveragePrices(u) = 0.0;
        for (Eigen::SparseMatrix<double>::InnerIterator c(problem.s_pod, u); c; ++c)
            vertexCoveragePrices(u) += coverageDuals(c.index()) * c.value();
    }
}

cost_t neighborhood_r::removalGain(int index) {
    auto current_segment = std::next(solution.path.begin(), index);
    cost_t gain = geometry.removal_gain(solution, current_segment);
    return gain;
}

cost_t neighborhood_r::replacementGain(int index, const segment &s) {
    cost_t cost{0,0};
    if (index == static_cast<int>(solution.path.size()) - 1)
        cost = geometry.insert_tail_gain(*std::prev(solution.path.end(), 2), s);
    else
        cost = geometry.insert_between_gain(*std::next(solution.path.begin(), index - 1),
                                               s,
                                               *std::next(solution.path.begin(),index + 1));
    return cost;
}

void neighborhood_r::generateColumn(PricingSolution &pricingSolution) {
    GRBColumn column;
    segment non_oriented = geometry.getNonOriented(pricingSolution.s);
    int u = non_oriented.source;
    int v = std::max(u, non_oriented.target);
    for (int n = u; n<= v; ++n) {
        for (Eigen::SparseMatrix<double>::InnerIterator c(problem.s_pod, n); c; ++c){
            if (solution.coverage(c.index()) < problem.req(c.index()))
                column.addTerm(c.value(), coverageConstraints[c.index()]);
        }
    }
    column.addTerm(1.0, reallocationConstraints[pricingSolution.R]);
    column.addTerm(pricingSolution.costs.length, pathLengthConstraint);
    reallocationSets[pricingSolution.R].variables.push_back(
            model->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, column, nullptr));
    reallocationSets[pricingSolution.R].moves.push_back({pricingSolution.s, pricingSolution.costs});
    *Z1 += pricingSolution.costs.length * reallocationSets[pricingSolution.R].variables.back();
    *Z2 += pricingSolution.costs.turns * reallocationSets[pricingSolution.R].variables.back();
}
