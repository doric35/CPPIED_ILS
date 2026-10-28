#pragma once

#include "construction.hpp"
#include "../neighborhoods/neighborhood_tsp.hpp"
#include "../algorithms/MaximumSubarrayKadane.hpp"
#include "../algorithms/MaximumSpaceConstrainedArraySelection.hpp"

namespace sweeper{
    struct SegmentSet{
        std::vector<segment> segments;
        std::vector<double> gains;
    };

    struct LexicographicSegmentSetValue{
        double gain;
        int negative_length;
        int negative_size;

        auto operator<=>(const LexicographicSegmentSetValue&) const = default;
    };
}

class DpSweeper: public construction{
protected:
    double mPenalty=0.01;

    std::vector<int> unsatisfiedCells;

    std::function<int(const std::array<sweeper::SegmentSet,2>&)> mChoice =
            [](const std::array<sweeper::SegmentSet,2>&){
        return -1;
    };

    void setRootSegmentIfEmptySolution();
    void setUnsatisfiedCells();
    std::array<sweeper::SegmentSet, 2> constructCandidateSets();
    void filterCandidateSetsWithSpacingConstraint(std::array<sweeper::SegmentSet, 2>& candidateSets);
    void filterCandidateSet(sweeper::SegmentSet& set);
    void selectCandidateSet(std::array<sweeper::SegmentSet, 2>& candidateSets);
    void addCoverageSegmentSet(const std::function<void(cppied_solution&)>& save_history);
    bool horizontalDominatesVertical(std::array<sweeper::SegmentSet, 2>& candidateSets);
    bool isValidChoice(int choice);
    void addSelectedSegmentsToSolution(const sweeper::SegmentSet& segments);
    void setInitialSolutionAlternateOrientations();
    void solveTspReduction();
    std::vector<double> constructArrayCosts(const segment& array);
    std::pair<int,int> getArrayBounds(const segment& array);

    std::pair<segment,double> maximumSubarray(const segment& array);

    sweeper::LexicographicSegmentSetValue getLexicographicValues(sweeper::SegmentSet& set);

public:
    template<typename... Args>
    explicit DpSweeper(Args&&... args) : construction(std::forward<Args>(args)...), unsatisfiedCells(){
        unsatisfiedCells.reserve(geometry.n_rows * geometry.n_cols);
    }

    void construct(const std::function<void(cppied_solution&)>& save_history) override;
    void set_choice(std::function<int(const std::array<sweeper::SegmentSet,2>&)> choice) {
        mChoice = std::move(choice);
    }
    [[nodiscard]] double get_penalty() const{return mPenalty;}
};