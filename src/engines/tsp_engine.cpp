#include "../../include/engines/tsp_engine.hpp"

std::vector<int> tsp_engine::lkh(Eigen::Ref<Eigen::MatrixXi> C,
                                 std::vector<int>& warm_start) {
    try {
        write_configuration(warm_start);
    } catch (timeout_error& e){
        std::cout << "Returning from LKH after encountering a timeout error." << std::endl;
        return warm_start;
    }
    write_instance_file(C);

    if (!warm_start.empty())
        write_initial_tour(warm_start);

    std::filesystem::path config_path = ctx.config.at("WORKING_DIRECTORY");
    std::filesystem::path config_name = config_path / (ctx.config.at("NAME") + ".par");
    std::filesystem::path sol_path = config_path / (ctx.config.at("NAME") + ".sol");
    std::filesystem::remove(sol_path);          // prevent stale-read

    std::string lkh_cmd = ctx.config.at("LKH_EXECUTABLE") + " " + config_name.string();
    std::string lkh_exec = lkh_cmd + " > /dev/null 2>&1";
    int result = std::system((lkh_exec).c_str());
    if (result != 0 || !std::filesystem::exists(sol_path))
        return warm_start;
    std::vector<int> tour;
    read_solution(tour);
    return tour;
}

void tsp_engine::write_configuration(std::vector<int> &warm_start) {
    std::ofstream config_stream;
    std::filesystem::path config_path = ctx.config.at("WORKING_DIRECTORY");
    std::filesystem::path config_name = config_path / (ctx.config.at("NAME") + ".par");
    config_stream.open(config_name.string());
    if (!config_stream.is_open())
        throw std::runtime_error("Could not open LKH configuration file for writting.");

    std::filesystem::path problem_name = config_path / (ctx.config.at("NAME") + ".tsp");
    std::filesystem::path sol_name = config_path / (ctx.config.at("NAME") + ".sol");
    config_stream << "PROBLEM_FILE = " << problem_name.string() << "\n";
    config_stream << "OUTPUT_TOUR_FILE = " << sol_name.string() << "\n";
    std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock ::now();
    int elapsed_time = int(std::chrono::duration_cast<std::chrono::seconds>(now - ctx.start_time).count());
    config_stream << "TIME_LIMIT = 10" << "\n";//<< ctx.max_time - elapsed_time << "\n";
    config_stream << "MAX_CANDIDATES = 6" << "\n";
    config_stream << "EXCESS = 1.0" << "\n";
    config_stream << "RECOMBINATION = CLARIST" << "\n";
    config_stream << "RESTRICTED_SEARCH = NO" << "\n";
    config_stream << "INITIAL_PERIOD = 100" << "\n";
    config_stream << "RUNS = 1\n";
    if ((ctx.max_time - elapsed_time) <= 0){
        config_stream.close();
        throw timeout_error("Did not run LKH due to time out.\n");
    }
    if (!warm_start.empty()){
        std::filesystem::path tour_name = config_path / (ctx.config.at("NAME") + ".tour");
        config_stream << "INITIAL_TOUR_FILE = " << tour_name.string() << "\n";
    }
    config_stream.close();
}

void tsp_engine::write_instance_file(Eigen::Ref<Eigen::MatrixXi> C) {
    std::ofstream problem_stream;
    std::filesystem::path problem_path = ctx.config.at("WORKING_DIRECTORY");
    std::filesystem::path config_name = problem_path / (ctx.config.at("NAME") + ".tsp");
    problem_stream.open(config_name.string());

    int dimension = static_cast<int>(C.rows());

    if (!problem_stream.is_open())
        throw std::runtime_error("Could not open LKH instance file for writting.");

    bool tsp = C.isApprox(C.transpose());
    std::string tsp_t = tsp ? "TSP" : "ATSP";

    problem_stream << "NAME : " << ctx.config.at("NAME") << "\n";
    problem_stream << "TYPE : " << tsp_t << "\n";
    problem_stream << "COMMENT : Sparse TSP with explicit edges\n";
    problem_stream << "DIMENSION : " << dimension << "\n";
    problem_stream << "EDGE_WEIGHT_TYPE : EXPLICIT\n";
    problem_stream << "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n\n";
    problem_stream << "EDGE_WEIGHT_SECTION\n";

    write_data(C, problem_stream);
    problem_stream << "EOF" << std::endl;
    problem_stream.close();
}

void tsp_engine::write_data(Eigen::Ref<Eigen::MatrixXi> C,
                                 std::ofstream &problem_stream) {
    std::ostringstream buffer;
    for (int i = 0; i < C.rows(); ++i) {
        for (int j = 0; j < C.cols(); ++j) {
            buffer << C(i, j);
            if (j < C.cols() - 1) buffer << ' ';
        }
        buffer << '\n';
    }

    problem_stream << buffer.str();
}

void tsp_engine::write_initial_tour(std::vector<int> &warm_start) {
    assert(!warm_start.empty());
    std::ofstream tour_stream;
    std::filesystem::path config_path = ctx.config.at("WORKING_DIRECTORY");
    std::filesystem::path tour_name = config_path / (ctx.config.at("NAME") + ".tour");
    tour_stream.open(tour_name.string());
    if (!tour_stream.is_open())
        throw std::runtime_error("Could not open LKH tour file for writting.");

    tour_stream << "NAME : " << ctx.config.at("NAME") << "\n";
    tour_stream << "TYPE : TOUR\n";
    tour_stream << "DIMENSION : " << warm_start.size() << "\n";
    tour_stream << "TOUR_SECTION\n";

    auto streamer = [&](int n){
        tour_stream << n << "\n";
    };
    std::for_each(warm_start.begin(), warm_start.end(), streamer);
    tour_stream << "-1" << std::endl;
    tour_stream.close();
}

void tsp_engine::read_solution(std::vector<int> &container) {
    std::ifstream tour_stream;
    std::filesystem::path config_path = ctx.config.at("WORKING_DIRECTORY");
    std::filesystem::path tour_name = config_path / (ctx.config.at("NAME") + ".sol");
    tour_stream.open(tour_name.string());
    if (!tour_stream.is_open())
        throw std::runtime_error("Could not open LKH tour file for reading.");

    std::string line;
    int dimension = -1;
    while (std::getline(tour_stream, line)){
        line = trim(line);
        if (line.empty() || line[0] == '#')
            continue;

        auto eq_pos = line.find(':');
        if (eq_pos == std::string::npos && line == "TOUR_SECTION") break;
        else {
            std::string key = trim(line.substr(0, eq_pos));
            std::string value = trim(line.substr(eq_pos+1));
            if (key == "DIMENSION") dimension = std::stoi(value);
        }
    }
    if (dimension == -1)
        throw std::runtime_error("Did not find dimension field in tsp sol file.");

    container.resize(dimension, 0);
    for (int i=0; i<dimension; i++)
        tour_stream >> container[i];
}

void tsp_engine::gtsp_to_atsp(tsp::GTSPGraph& graph) {
    gtsp_to_sgtsp(graph);
    sgtsp_to_atsp(graph);
}

void tsp_engine::gtsp_to_sgtsp(tsp::GTSPGraph& graph) {
    Eigen::MatrixXi C_hat(graph.cost.rows(), graph.cost.cols());
    C_hat.setConstant(graph.penalty);
    // Create the intra-cluster loops
    for (auto & cluster : graph.clusters){
        assert(!cluster.nodes.empty());
        for (int i=0; i< cluster.nodes.size()-1;++i){
            C_hat(cluster.nodes[i],
                  cluster.nodes[i+1]) = 0;
        }
        if (cluster.nodes.size() > 1)
            C_hat(cluster.nodes.back(),
                  cluster.nodes.front()) = 0;
    }
    //Create the inter-cluster edges
    for (int i=0; i< graph.clusters.size(); ++i) {
        for (int k=0; k<graph.clusters.size(); k++){
            if (k!= i){
                for (int j=1; j< graph.clusters[i].nodes.size(); ++j){
                    for (int k_l: graph.clusters[k].nodes){
                        C_hat(graph.clusters[i].nodes[j-1],
                              k_l) = graph.cost(graph.clusters[i].nodes[j],
                                                k_l);
                    }
                }
                for (int k_l: graph.clusters[k].nodes) {
                    C_hat(graph.clusters[i].nodes.back(),
                          k_l) = graph.cost(graph.clusters[i].nodes.front(), k_l);
                }
            }
        }
    }
    graph.cost.swap(C_hat);
}

void tsp_engine::sgtsp_to_atsp(tsp::GTSPGraph& graph) {
//    int64_t sum64 = 0;
//    for (int r = 0; r < graph.cost.rows(); ++r)
//        for (int c = 0; c < graph.cost.cols(); ++c)
//            if (graph.cost(r, c) != graph.penalty)
//                sum64 += graph.cost(r, c);
//    sum64 += 1;
//    if (static_cast<int64_t>(graph.penalty) == sum64)
//        sum64 += 1;
//    int sum = static_cast<int>(
//        std::min(sum64, static_cast<int64_t>(std::numeric_limits<int>::max() / 2)));

//    graph.cost.array() = graph.cost.array() + sum;
//
//    for (auto& c : graph.clusters){
//        for (int i=0; i< c.nodes.size() -1; ++i)
//            graph.cost(c.nodes[i], c.nodes[i+1]) = 0;
//
//        if (c.nodes.size() > 1)
//            graph.cost(c.nodes.back(), c.nodes.front()) = 0;
//    }
    //Identity transformation, the penalty must have been already given.
}

void tsp_engine::tsp_tour_to_gtsp_tour(std::vector<int> &tour,
                                       std::vector<std::shared_ptr<tsp::cluster>>& map) {
    std::vector<int> gtsp_tour;
    gtsp_tour.reserve(tour.size());
    auto n1 = tour.begin();
    auto n2 = std::next(tour.begin());
    for (; n1 != tour.end(); n1 = n2){
        gtsp_tour.push_back(*n1);
        for (; n2 != tour.end() && map[*n2] == map[*n1]; ++n2)
            continue;
    }
    tour.swap(gtsp_tour);
}



