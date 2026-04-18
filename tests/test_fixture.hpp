#pragma once

#include <gtest/gtest.h>
#include "../include/structures.hpp"
#include "../include/cppied_method_base.hpp"
#include "../include/utils.hpp"

struct cppied_context_fixture: public ::testing::Test{
protected:
    cppied_context_fixture(): ctx(read_configuration(config_path)),
                              P(read_matrix<int>(seabed_path),
                                  read_matrix<double>(pod_path),
                                  read_matrix<double>(req_path)){}
    cppied_context ctx;
    cppied_instance P;
    // Raw data files live in the source tree
    static constexpr const char* seabed_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_seabed.txt";
    static constexpr const char* pod_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_pod.txt";
    static constexpr const char* req_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_req.txt";
    // Generated config files (with resolved paths) live in the build tree
    static constexpr const char* config_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_config.txt";
};

struct cppied_method_fixture: public cppied_context_fixture{
protected:
    struct derived_method : public cppied_method_base {
        explicit derived_method(cppied_context& context, cppied_instance& instance):
            cppied_method_base(context, instance){}
        void initialize() override{}
        void terminate() override{}
        void d_solve(cppied_solution& pSol) override{}
        using cppied_method_base::geometry;
        using cppied_method_base::coverage;
    };

    cppied_method_fixture():cppied_context_fixture(), method_instance(ctx, P){
    }

    derived_method method_instance;
};

struct cppied_context_large_fixture: public ::testing::Test{
protected:
    cppied_context_large_fixture(): ctx(read_configuration(config_path)),
                              P(read_matrix<int>(seabed_path),
                                read_matrix<double>(pod_path),
                                read_matrix<double>(req_path)){}
    cppied_context ctx;
    cppied_instance P;
    // Raw data files live in the source tree
    static constexpr const char* seabed_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_s1616_ir0_lrc031/cppied_problem.txt";
    static constexpr const char* pod_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_s1616_ir0_lrc031/cppied_pod.txt";
    static constexpr const char* req_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_s1616_ir0_lrc031/cppied_req.txt";
    // Generated config files (with resolved paths) live in the build tree
    static constexpr const char* config_path =
            PROJECT_SOURCE_DIR "/tests/configurations/ilp_test_config.txt";
};