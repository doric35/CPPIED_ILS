#include "include/utils.hpp"
#include "include/structures.hpp"
#include "include/solver_factory.hpp"
#include "include/metaheuristics/ils.hpp"
#include "include/models/ilp.hpp"

REGISTER_SOLVER("ILS", ils)
REGISTER_SOLVER("GUROBI", ilp)

class viz_helper : public cppied_method_base{
public:
    using cppied_method_base::cppied_method_base;
    using cppied_method_base::coverage;
    using cppied_method_base::geometry;
    using cppied_method_base::problem;

    void initialize() override{
        throw std::runtime_error("Method should not be called for helper.");
    };
    void terminate() override{
        throw std::runtime_error("Method should not be called for helper.");
    };
    void d_solve(cppied_solution& pSolution) override{
        throw std::runtime_error("Method should not be called for helper.");
    };
};

int main(int argc, char* argv[]) {
    std::cout << "Hello, welcome to CPPIED heuristic methods program." << std::endl;
    std::cout << "Setting up..." << std::endl;
    if (argc != 2){
        std::cerr << "Single argument required. Give the path to the configuration file." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::unordered_map<std::string, std::string> configuration;
    try {
        configuration = read_configuration(argv[1]);
    } catch (std::exception& e){
        std::cerr << "\nError while reading configuration file: \n" << e.what() << std::endl;
        return 1;
    }

    std::set<VerboseFlag> verbose_flags;
    if (configuration.count("VERBOSE"))
        verbose_flags = parse_verbose(configuration["VERBOSE"]);

    MatrixXdRow<int> seabed;
    MatrixXdRow<double> req_coverage;
    MatrixXdRow<double> pod;
    try{
        seabed = read_matrix<int>(configuration.at("SEABED_FILE"));
        req_coverage = read_matrix<double>(configuration.at("REQ_COVERAGE_FILE"));
        pod = read_matrix<double>(configuration.at("POD_FILE"));
    } catch  (std::exception& e){
        std::cerr << "\nError while reading the CPPIED instance: \n" << e.what() << std::endl;
        return 1;
    }
    if (seabed.rows() != req_coverage.rows() ||
        seabed.cols() != req_coverage.cols()) {
        throw std::runtime_error("Seabed and required coverage dimensions mismatch.");
    }

    cppied_instance instance(seabed, pod, req_coverage);
    cppied_solution solution = instance.initialize_solution();

    cppied_context ctx(configuration);

    std::cout << "Available solvers: ";
    SolverFactory::instance().debug_print();
    auto solver = SolverFactory::instance().create(configuration.at("SOLVER"), ctx, instance);
    if (!solver)
        throw std::runtime_error("Solver creation failed for: " + configuration.at("SOLVER"));

    if (verbose_flags.count(VerboseFlag::DEBUG)) {
        CPPIEDCallbacks cb{};
        cb.onSatisfy = [](const cppied_solution& sol, const std::any& a){
            assert(true);
        };
        solver->set_callbacks(cb);
    }
    std::cout << "Starting the optimization algorithm with solver " << configuration.at("SOLVER") << std::endl;
    solver -> solve(solution);

    if (verbose_flags.count(VerboseFlag::SUMMARY))
        write_csv(configuration.at("CSV_LOG_FILE"), ctx.csv_log);

    if (verbose_flags.count(VerboseFlag::SOLUTION)){
        std::map<std::string, std::string> output_header;
        output_header["NAME"] = configuration.at("NAME");
        output_header["COMMENT"] = "Congratulations, the program ran without with a runtime error.";
        output_header["ALGORITHM_CONFIGURATION" ] = configuration.at("ALGORITHM_CONFIG");
        write_solution(configuration.at("SOLUTION_FILE"), output_header, solution);
    }

    if (verbose_flags.count(VerboseFlag::VIZ)){
        viz_helper h(ctx, instance);
        std::string viz_file = configuration.at("VIZ_FILE");
        std::string png_file = viz_file.substr(
                0,
               viz_file.find('.')
        );
        png_file += ".png";
        write_visulization(configuration.at("VIZ_FILE"),
                           png_file, solution,
                           h, 10);
    }
    return 0;
}