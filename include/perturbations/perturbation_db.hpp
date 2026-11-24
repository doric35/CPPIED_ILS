#pragma once

#include "perturbation.hpp"

class perturbation_db : public perturbation{
public:
    using perturbation::perturbation;

    void d_perturbate() override;

    static bool isValidSolutionInstance(const cppied_solution& solution){
        bool is_valid = solution.path.size() >= 10;
        return is_valid;
    }
protected:

    std::array<int, 4> getDoubleBridgeIndexes();
    std::vector<int> uniformCandidateSample();
    int greedyCandidateSample(std::vector<int>& candidates);

    std::array<int, 6> topSixCandidates(std::vector<int>& candidates, int ref);
    std::vector<cost_t> getDistancesToReferenceCandidate(std::vector<int>& candidates, int ref);
    std::vector<int> getOrderedIndexes(std::vector<cost_t>& distances);

    void applyDoubleBridgeMove(std::array<int, 4>& selection);

    cost_t gain(std::vector<int>& sampling, int i);
};