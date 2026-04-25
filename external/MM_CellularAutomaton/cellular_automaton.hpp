#pragma once

#include "utils.hpp"

namespace generation_algorithms {
    struct automaton {
        unsigned seed;
        unsigned W;
        unsigned H;

        const std::vector<unsigned>& alphabet;
        std::vector<std::vector<unsigned>> mCells;

        std::mt19937 rng;
        automaton(unsigned pseed, unsigned pW,
                  unsigned pH, const std::vector<unsigned>& palphabet) :
                seed(pseed), W(pW), H(pH), alphabet(palphabet), mCells(), rng(seed){
            mCells.resize(H, std::vector<unsigned>(H, 0));
        }
        virtual void run() = 0;
        virtual ~automaton() = default;
    };

    struct uniform_automaton : automaton{
        using automaton::automaton;
        void run() override{}
    };

    inline std::unique_ptr<automaton> make_algorithm(algorithm_type t) {
        switch (t) {
            case algorithm_type::UNIFORM:
                return std::make_unique<uniform_automaton>();
            case algorithm_type::CELLULAR_AUTOMATON:
                return std::make_unique<CellularAutomaton>();
        }
        throw std::invalid_argument("Unknown algorithm_type");
    }

    inline std::unique_ptr<automaton> make_algorithm(const std::string& sT) {
        algorithm_type t = to_type(sT);
        switch (t) {
            case algorithm_type::UNIFORM:
                return std::make_unique<uniform_automaton>();
            case algorithm_type::CELLULAR_AUTOMATON:
                return std::make_unique<cellular_automaton>();
        }
        throw std::invalid_argument("Unknown algorithm_type");
    }
}

class cellular_automaton {
public:
    cellular_automaton(unsigned seed, unsigned W, unsigned H, const std::vector<unsigned>& alphabet) :
        mSeed(seed), mW(W), mH(H), mAlphabet(alphabet), mCells(), rng(seed){
        mCells.resize(mH, std::vector<unsigned>(mH, 0));
    }

    void generate();

    bool game_of_life();
    bool four_five();
    bool game_of_life_health_based();

    void dead_or_alive_random_pattern();
    void uniform_random_pattern();

    const std::vector<std::vector<unsigned>>& get_cells(){
        return mCells;
    }
    friend generation_algorithms::automaton;

protected:
    unsigned mSeed;
    unsigned mW;
    unsigned mH;

    std::vector<unsigned>              mAlphabet;
    std::vector<std::vector<unsigned>> mCells;

    std::mt19937 rng;
};