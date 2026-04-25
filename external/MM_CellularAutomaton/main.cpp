#include "utils.hpp"
#include "cellular_automaton.hpp"

int main (int argc, char **argv){
    std::cout << "Welcome to the CPPIED instances seabed generator." << std::endl;
    if (argc < 2)
    {
        std::cerr << "[ERROR]: Give the absolute path to the configuration file as input." << std::endl;
        return -1;
    }

    std::string configuration_filename = argv[1];
    std::cout << "Using configuration file:\n" << configuration_filename << std::endl;

    std::map<std::string, std::string> configuration;
    read_configuration(configuration_filename, configuration);


    return 0;
}