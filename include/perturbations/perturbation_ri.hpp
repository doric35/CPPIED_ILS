#pragma once

#include "perturbation.hpp"
#include "../algorithms/MinimumSegmentInsertion.hpp"

class perturbation_ri : public perturbation{
public:
    template<typename... Args>
    explicit perturbation_ri(Args&&... args) : perturbation(std::forward<Args>(args)...),
        dist(0.0, 1.0), RATIO(0.05),
        sampleSize(static_cast<size_t>(std::ceil(RATIO*double(solution.path.size())))){}

    void d_perturbate() override;

protected:
    const double RATIO = 0.05;
    std::size_t sampleSize;
    std::vector<segment> insertionSample;

    void sampleCandidateUnitarySegments();
    std::vector<double> getCandidateWeights();
    double sampleCandidateWeight(int u);

    [[nodiscard]] std::vector<int> getSortedCandidates(const std::vector<double>& weights) const;
    void setCandidateSampling(const std::vector<int>& partiallySortedCandidates);

    void insertCandidates();
    void insertCandidate(segment& s);

private:
    std::uniform_real_distribution<> dist;
};