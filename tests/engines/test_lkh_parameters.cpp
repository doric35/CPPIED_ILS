#include "../test_fixture.hpp"
#include "../../include/engines/tsp_engine.hpp"

// ============================================================================
// LKH parameters read from an external file.
//
//   cppied_context::set_lkh_params (called by the constructor)
//     - LKH_PARAMETERS is optional in the configuration; when absent it is set
//       to PROJECT_ROOT_DIR/include/metadata/lkh_parameters.txt.
//     - the file is read with read_configuration into ctx.lkh_params.
//     - on a read error, the path is printed and the exception rethrown.
//   tsp_engine::write_lkh_params
//     - writes every "KEY = VALUE" pair of ctx.lkh_params.
//   tsp_engine::write_configuration
//     - writes PROBLEM_FILE, OUTPUT_TOUR_FILE, the LKH parameters and, with a
//       warm start, INITIAL_TOUR_FILE into WORKING_DIRECTORY/NAME.par.
// ============================================================================

namespace {

namespace fs = std::filesystem;
using config_t = std::unordered_map<std::string, std::string>;

// The values hard-coded in write_configuration before this change.
const config_t previous_hard_coded_params = {
    {"TIME_LIMIT", "10"},
    {"MAX_CANDIDATES", "6"},
    {"EXCESS", "1.0"},
    {"RECOMBINATION", "CLARIST"},
    {"RESTRICTED_SEARCH", "NO"},
    {"INITIAL_PERIOD", "100"},
    {"RUNS", "1"},
};

fs::path default_params_path() {
    return fs::path(PROJECT_ROOT_DIR) / "include" / "metadata" / "lkh_parameters.txt";
}

// Temporary directory removed at the end of each test.
class lkh_params_fixture : public ::testing::Test {
protected:
    fs::path dir;

    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        dir = fs::temp_directory_path() /
              (std::string("cppied_lkh_") + info->name());
        fs::remove_all(dir);
        fs::create_directories(dir);
    }
    void TearDown() override { fs::remove_all(dir); }

    fs::path write_file(const std::string& name, const std::string& content) {
        fs::path p = dir / name;
        std::ofstream(p) << content;
        return p;
    }

    static config_t base_config() { return {{"TIME", "600"}}; }
};

} // namespace

// ---------------------------------------------------------------------------
// cppied_context::set_lkh_params — default file
// ---------------------------------------------------------------------------

TEST_F(lkh_params_fixture, DefaultPathIsSetWhenKeyIsMissing) {
    cppied_context ctx(base_config());
    ASSERT_TRUE(ctx.config.contains("LKH_PARAMETERS"));
    EXPECT_EQ(fs::path(ctx.config.at("LKH_PARAMETERS")).lexically_normal(),
              default_params_path().lexically_normal());
}

TEST_F(lkh_params_fixture, DefaultFileExists) {
    EXPECT_TRUE(fs::exists(default_params_path())) << default_params_path();
}

// Moving the parameters to a file must not change LKH's behaviour.
TEST_F(lkh_params_fixture, DefaultParametersMatchPreviousHardCodedValues) {
    cppied_context ctx(base_config());
    EXPECT_EQ(ctx.lkh_params, previous_hard_coded_params);
}

// The test configurations have no LKH_PARAMETERS key: every existing fixture
// now runs with the default file.
TEST_F(lkh_params_fixture, TestConfigurationGetsDefaultParameters) {
    cppied_context ctx(read_configuration(
        PROJECT_SOURCE_DIR "/tests/configurations/test_config.txt"));
    EXPECT_EQ(ctx.lkh_params, previous_hard_coded_params);
}

// RunConfiguration (main.cpp) and the tests copy contexts.
TEST_F(lkh_params_fixture, CopiedContextKeepsParameters) {
    cppied_context ctx(base_config());
    cppied_context copy = ctx;
    EXPECT_EQ(copy.lkh_params, ctx.lkh_params);
    EXPECT_EQ(copy.config.at("LKH_PARAMETERS"), ctx.config.at("LKH_PARAMETERS"));
}

// ---------------------------------------------------------------------------
// cppied_context::set_lkh_params — explicit file
// ---------------------------------------------------------------------------

TEST_F(lkh_params_fixture, ExplicitFileReplacesDefault) {
    fs::path p = write_file("custom.par", "RUNS = 3\nMAX_TRIALS = 50\n");
    config_t config = base_config();
    config["LKH_PARAMETERS"] = p.string();
    cppied_context ctx(config);
    EXPECT_EQ(ctx.config.at("LKH_PARAMETERS"), p.string());
    EXPECT_EQ(ctx.lkh_params, (config_t{{"RUNS", "3"}, {"MAX_TRIALS", "50"}}));
}

TEST_F(lkh_params_fixture, CommentsBlankLinesAndSpacesAreIgnored) {
    fs::path p = write_file("custom.par",
                            "# LKH settings\n\n   RUNS   =   2   \n\t\n# end");
    config_t config = base_config();
    config["LKH_PARAMETERS"] = p.string();
    cppied_context ctx(config);
    EXPECT_EQ(ctx.lkh_params, (config_t{{"RUNS", "2"}}));
}

// No trailing newline on the last line (as in the default file).
TEST_F(lkh_params_fixture, LastLineWithoutNewlineIsRead) {
    fs::path p = write_file("custom.par", "RUNS = 1\nSEED = 7");
    config_t config = base_config();
    config["LKH_PARAMETERS"] = p.string();
    cppied_context ctx(config);
    EXPECT_EQ(ctx.lkh_params.at("SEED"), "7");
}

// An empty file is accepted: LKH then falls back to its own defaults
// (e.g. RUNS = 10, no TIME_LIMIT).
TEST_F(lkh_params_fixture, EmptyFileGivesNoParameters) {
    fs::path p = write_file("empty.par", "");
    config_t config = base_config();
    config["LKH_PARAMETERS"] = p.string();
    cppied_context ctx(config);
    EXPECT_TRUE(ctx.lkh_params.empty());
}

// ---------------------------------------------------------------------------
// cppied_context::set_lkh_params — errors
// ---------------------------------------------------------------------------

// The original exception type and message must reach the caller.
TEST_F(lkh_params_fixture, MissingFileThrowsRuntimeErrorWithMessage) {
    config_t config = base_config();
    config["LKH_PARAMETERS"] = (dir / "does_not_exist.par").string();
    try {
        cppied_context ctx(config);
        FAIL() << "no exception thrown";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("Unable to open"), std::string::npos)
            << e.what();
    } catch (const std::exception& e) {
        FAIL() << "exception type lost (sliced to std::exception): " << e.what();
    }
}

TEST_F(lkh_params_fixture, MalformedLineThrowsRuntimeErrorWithMessage) {
    fs::path p = write_file("bad.par", "RUNS = 1\nMAX_TRIALS 50\n");
    config_t config = base_config();
    config["LKH_PARAMETERS"] = p.string();
    try {
        cppied_context ctx(config);
        FAIL() << "no exception thrown";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("line 2"), std::string::npos)
            << e.what();
    } catch (const std::exception& e) {
        FAIL() << "exception type lost (sliced to std::exception): " << e.what();
    }
}

// A path relative to the working directory of the process.
TEST_F(lkh_params_fixture, RelativePathIsResolvedFromCurrentDirectory) {
    write_file("relative.par", "RUNS = 4\n");
    fs::path old = fs::current_path();
    fs::current_path(dir);
    config_t config = base_config();
    config["LKH_PARAMETERS"] = "relative.par";
    EXPECT_NO_THROW({
        cppied_context ctx(config);
        EXPECT_EQ(ctx.lkh_params.at("RUNS"), "4");
    });
    fs::current_path(old);
}

// ---------------------------------------------------------------------------
// tsp_engine::write_lkh_params
// ---------------------------------------------------------------------------

TEST_F(lkh_params_fixture, WriteLkhParamsRoundTrips) {
    cppied_context ctx(base_config());
    tsp_engine engine(ctx);
    fs::path p = dir / "out.par";
    {
        std::ofstream out(p);
        engine.write_lkh_params(out);
    }
    EXPECT_EQ(read_configuration(p.string()), ctx.lkh_params);
}

TEST_F(lkh_params_fixture, WriteLkhParamsWritesOneLinePerParameter) {
    cppied_context ctx(base_config());
    tsp_engine engine(ctx);
    fs::path p = dir / "out.par";
    {
        std::ofstream out(p);
        engine.write_lkh_params(out);
    }
    std::ifstream in(p);
    std::string line;
    std::size_t n = 0;
    while (std::getline(in, line)) {
        ++n;
        EXPECT_NE(line.find(" = "), std::string::npos) << line;
    }
    EXPECT_EQ(n, ctx.lkh_params.size());
}

TEST_F(lkh_params_fixture, WriteLkhParamsWithNoParameterWritesNothing) {
    fs::path empty = write_file("empty.par", "");
    config_t config = base_config();
    config["LKH_PARAMETERS"] = empty.string();
    cppied_context ctx(config);
    tsp_engine engine(ctx);
    fs::path p = dir / "out.par";
    {
        std::ofstream out(p);
        engine.write_lkh_params(out);
    }
    EXPECT_EQ(fs::file_size(p), 0u);
}

// ---------------------------------------------------------------------------
// tsp_engine::write_configuration
// ---------------------------------------------------------------------------

class lkh_write_configuration_fixture : public lkh_params_fixture {
protected:
    config_t config() {
        config_t c = base_config();
        c["NAME"] = "lkh_params_test";
        c["WORKING_DIRECTORY"] = dir.string();
        return c;
    }
    config_t par_file() {
        return read_configuration((dir / "lkh_params_test.par").string());
    }
};

TEST_F(lkh_write_configuration_fixture, ContainsEngineFilesAndEveryParameter) {
    cppied_context ctx(config());
    ctx.start_time = std::chrono::high_resolution_clock::now();
    tsp_engine engine(ctx);
    std::vector<int> no_warm_start;
    engine.write_configuration(no_warm_start);

    config_t par = par_file();
    EXPECT_EQ(par.at("PROBLEM_FILE"), (dir / "lkh_params_test.tsp").string());
    EXPECT_EQ(par.at("OUTPUT_TOUR_FILE"), (dir / "lkh_params_test.sol").string());
    EXPECT_FALSE(par.contains("INITIAL_TOUR_FILE"));
    for (auto& [k, v] : ctx.lkh_params)
        EXPECT_EQ(par.at(k), v) << k;
}

TEST_F(lkh_write_configuration_fixture, WarmStartAddsInitialTourFile) {
    cppied_context ctx(config());
    ctx.start_time = std::chrono::high_resolution_clock::now();
    tsp_engine engine(ctx);
    std::vector<int> warm_start = {0, 1, 2};
    engine.write_configuration(warm_start);
    EXPECT_EQ(par_file().at("INITIAL_TOUR_FILE"),
              (dir / "lkh_params_test.tour").string());
}

// The parameter file is written before the timeout check: on timeout the
// call must still throw timeout_error (lkh() relies on it).
TEST_F(lkh_write_configuration_fixture, TimeoutStillThrows) {
    config_t c = config();
    c["TIME"] = "0";
    cppied_context ctx(c);
    ctx.start_time = std::chrono::high_resolution_clock::now();
    tsp_engine engine(ctx);
    std::vector<int> no_warm_start;
    EXPECT_THROW(engine.write_configuration(no_warm_start), timeout_error);
}

// The files managed by the engine must not be redirected by the parameter
// file. LKH keeps the last value of a repeated keyword, as read_configuration
// does here.
TEST_F(lkh_write_configuration_fixture, ParametersCannotOverrideEngineFiles) {
    fs::path p = write_file("custom.par",
                            "RUNS = 1\n"
                            "PROBLEM_FILE = /elsewhere.tsp\n"
                            "OUTPUT_TOUR_FILE = /elsewhere.sol\n");
    config_t c = config();
    c["LKH_PARAMETERS"] = p.string();
    cppied_context ctx(c);
    ctx.start_time = std::chrono::high_resolution_clock::now();
    tsp_engine engine(ctx);
    std::vector<int> no_warm_start;
    engine.write_configuration(no_warm_start);

    config_t par = par_file();
    EXPECT_EQ(par.at("PROBLEM_FILE"), (dir / "lkh_params_test.tsp").string());
    EXPECT_EQ(par.at("OUTPUT_TOUR_FILE"), (dir / "lkh_params_test.sol").string());
}
