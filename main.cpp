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



class RunConfiguration{
protected:
    void set_solver(){
        solver = SolverFactory::instance().create(configuration.at("SOLVER"), context, instance);
        if (!solver)
            throw std::runtime_error("Solver creation failed for: " + configuration.at("SOLVER"));
    }
    void set_debug_callbacks(){
        if (!flags.count(VerboseFlag::DEBUG))
            return;
        CPPIEDCallbacks cb{};
        cb.onSatisfy = [](const cppied_solution& sol, const std::any& a){
            assert(true);
        };
        solver->set_callbacks(cb);
    }
public:
    // The solver holds references to context and instance: never copy or move.
    RunConfiguration(const RunConfiguration&) = delete;
    RunConfiguration& operator=(const RunConfiguration&) = delete;
    RunConfiguration(std::unordered_map<std::string, std::string>& config, cppied_context& ctx, cppied_instance& i,
                     cppied_solution& s, std::set<VerboseFlag>& f) : configuration(config), context(ctx),
                                                                     instance(i), solution(s), flags(f), solver(){
        set_solver();
        set_debug_callbacks();
    }
    std::unordered_map<std::string, std::string> configuration;
    cppied_context context;
    cppied_instance instance;
    cppied_solution solution;
    std::set<VerboseFlag> flags;
    std::unique_ptr<cppied_method_base> solver;
};

void print_global_information(){
    std::cout << "Hello, welcome to CPPIED heuristic methods program." << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "Available solvers : \n";
    SolverFactory::instance().debug_print();
    std::cout << "------------------------------" << std::endl;
    std::cout << "Setting up..." << std::endl;
}

void print_run_information(const RunConfiguration& runConfiguration){
    std::cout << "------------------------------" << std::endl;
    std::cout << "Name : " << runConfiguration.configuration.at("NAME") << std::endl;
    std::cout << "Algorithm configuration : " << runConfiguration.configuration.at("ALGORITHM_CONFIG") << std::endl;
    std::cout << "Solver : " << runConfiguration.configuration.at("SOLVER") << std::endl;
    std::cout << "------------------------------" << std::endl;
}

void post_solve_information(RunConfiguration& runConfiguration){
    if (runConfiguration.flags.count(VerboseFlag::SUMMARY))
        write_csv(runConfiguration.configuration.at("CSV_LOG_FILE"), runConfiguration.context.csv_log);

    if (runConfiguration.flags.count(VerboseFlag::SOLUTION)){
        std::map<std::string, std::string> output_header;
        output_header["NAME"] = runConfiguration.configuration.at("NAME");
        output_header["COMMENT"] = "Congratulations, the program ran without with a runtime error.";
        output_header["ALGORITHM_CONFIGURATION" ] = runConfiguration.configuration.at("ALGORITHM_CONFIG");
        write_solution(runConfiguration.configuration.at("SOLUTION_FILE"), output_header, runConfiguration.solution);
    }

    if (runConfiguration.flags.count(VerboseFlag::VIZ)){
        viz_helper h(runConfiguration.context, runConfiguration.instance);
        std::string viz_file = runConfiguration.configuration.at("VIZ_FILE");
        std::string png_file = viz_file.substr(
                0,
                viz_file.find('.')
        );
        std::string sol_frame_fn = png_file + "_sol" + ".png";
        png_file += ".png";
        write_visulization(runConfiguration.configuration.at("VIZ_FILE"),
                           png_file, sol_frame_fn, runConfiguration.solution,
                           h, 10);
    }
}

bool isValidArguments(int argc, char* argv[]){
    bool is_valid = true;
    if (argc != 2) {
        std::cerr << "Single argument required. Give the path to the configuration file." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        is_valid = false;
    } else {
        std::filesystem::path filename = argv[1];
        if (!std::filesystem::exists(filename)) {
            std::cerr << "Configuration file <" << filename.string() << "> does not exists." << std::endl;
            is_valid = false;
        }
    }
    return is_valid;
}

std::unordered_map<std::string, std::string> get_configuration(const std::string& filename){
    std::unordered_map<std::string, std::string> configuration;
    try {
        configuration = read_configuration(filename);
    } catch (std::exception& e){
        std::cerr << "\nError while reading configuration file: \n" << e.what() << std::endl;
        configuration = {};
    }
    return configuration;
}

cppied_instance get_instance(const std::unordered_map<std::string, std::string>& configuration){
    MatrixXdRow<int> seabed;
    MatrixXdRow<double> req_coverage;
    MatrixXdRow<double> pod;
    try{
        seabed = read_matrix<int>(configuration.at("SEABED_FILE"));
        req_coverage = read_matrix<double>(configuration.at("REQ_COVERAGE_FILE"));
        pod = read_matrix<double>(configuration.at("POD_FILE"));
    } catch  (std::exception& e){
        std::cerr << "\nError while reading the CPPIED instance: \n" << e.what() << std::endl;
        throw;
    }
    if (seabed.rows() != req_coverage.rows() || seabed.cols() != req_coverage.cols())
        throw std::runtime_error("Seabed and required coverage dimensions mismatch.");

    return cppied_instance(seabed, pod, req_coverage);
}

std::unique_ptr<RunConfiguration> get_run_configuration(const std::filesystem::path& filename){
    std::unordered_map<std::string, std::string> configuration = get_configuration(filename.string());

    std::set<VerboseFlag> verbose_flags;
    if (configuration.count("VERBOSE"))
        verbose_flags = parse_verbose(configuration["VERBOSE"]);

    cppied_instance instance = get_instance(configuration);
    cppied_solution solution = instance.initialize_solution();
    cppied_context ctx(configuration);

    return std::make_unique<RunConfiguration>(configuration, ctx, instance, solution, verbose_flags);
}

int main(int argc, char* argv[]) {
    print_global_information();
    if (!isValidArguments(argc, argv)) return 1;
    std::unique_ptr<RunConfiguration> run_config;
    try {
        run_config = get_run_configuration(std::filesystem::path(argv[1]));
    } catch (std::exception& e){
        std::cerr << "[ERROR] : " << e.what() << std::endl;
        return 1;
    }
    print_run_information(*run_config);
    run_config->solver->solve(run_config->solution);
    post_solve_information(*run_config);
    return 0;
}