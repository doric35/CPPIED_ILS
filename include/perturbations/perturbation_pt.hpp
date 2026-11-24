#pragma once

#include "perturbation.hpp"
#include "../neighborhoods/neighborhood_trim.hpp"

class perturbation_pt : public perturbation{
public:
    template<typename... Args>
    perturbation_pt(Args&&... args) : perturbation(std::forward<Args>(args)...), RATIO(0.05),
        K(static_cast<std::size_t>(std::ceil(RATIO*double(solution.path.size())))){}

    void d_perturbate() override;

protected:
    const double RATIO;
    std::size_t K;

    std::vector<int> sampleCandidates();
    void replaceCandidates(const std::vector<int>& candidates);
    void replaceSegment(std::vector<segment>& new_sol, int i);
    std::vector<segment> getReplacements(const segment& s);

    std::vector<segment> getHorizontalCoverReplacements(const iRectangle& box);
    std::vector<segment> getVerticalCoverReplacements(const iRectangle& box);

    void setReplacementsCoverage(const std::vector<segment>& replacementSet);
    direction getReplacementDirection(const std::vector<segment>& new_sol,
                                      const std::vector<segment>& replacementSet);
    void pushBackParallelTracks(std::vector<segment>& new_sol,
                                const std::vector<segment>& replacementSet,
                                direction firstDirection);
};