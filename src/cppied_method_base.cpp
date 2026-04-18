#include "../include/cppied_method_base.hpp"

void cppied_method_base::solve(cppied_solution &solution) {
    assert(solution.coverage.size() == problem.cells.cols());
    initialize();
    ctx.start_time = std::chrono::high_resolution_clock::now();
    std::string status;
    bool validate = true;
    try {
        d_solve(solution);
    } catch (algo_flag& f){
        if (f.flag() == algorithm_flag::TIME_LIMIT_INFEASIBLE){
            std::cerr << "[WARNING] could not find a feasible solution within allowed time." << std::endl;
            status = "TIME_LIMIT_INFEASIBLE";
            validate = false;
        } else if (f.flag() == algorithm_flag::OPTIMAL)
            status = "OPTIMAL";
        else
            status = "SUBOPTIMAL";
    } catch (std::runtime_error& e){
        std::cerr << e.what() << std::endl;
        throw e;
    }
    ctx.end_time = std::chrono::high_resolution_clock::now();
    if (validate)
        validate_solution(solution);
    terminate();
    ctx.csv_log.push_back(ctx.config["NAME"]);
    ctx.csv_log.push_back(ctx.config["SOLVER"]);
    ctx.csv_log.push_back(std::to_string(solution.cost.length));
    ctx.csv_log.push_back(std::to_string(solution.cost.turns));
    auto elapsed_time =
            std::chrono::duration_cast<std::chrono::seconds>(ctx.end_time - ctx.start_time);
    ctx.csv_log.push_back(std::to_string(elapsed_time.count()));
    ctx.csv_log.push_back(status);
}

void cppied_method_base::validate_solution(cppied_solution &pSolution) {
    if ((pSolution.coverage.array() < problem.req.array()).any())
        throw algo_exception("Coverage constraints not satisfied");

    if (Point2F::L1(problem.vertex[pSolution.path[0].source],problem.vertex[problem.initial_position]) >= 1e-5)
        throw algo_exception("Initial position constraint not satisfied");

    for (size_t v = 0; v + 1 < pSolution.path.size(); ++v){
        if (!geometry.satisfy(pSolution.path[v], pSolution.path[v+1]))
            throw algo_exception("Path constraints not satisfied");
    }

    cost_t c = pSolution.cost;
    if (c != geometry.cost(pSolution))
        throw algo_exception("Wrong path length computation");

    int turn_count = int(pSolution.path.size()) - 1; //Segments are separated by turns here.
    if (pSolution.cost.turns != turn_count)
        throw algo_exception("Wrong path length computation");
}

int cppied_method_base::remaining_time() {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed = current_time - ctx.start_time;
    int elapsed_sec = static_cast<int>(
            std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
    );
    return ctx.max_time - elapsed_sec;
}