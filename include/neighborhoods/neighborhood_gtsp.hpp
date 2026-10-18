#pragma once

#include "neighborhood.hpp"
#include "../engines/tsp_engine.hpp"
#include "neighborhood_trim.hpp"
#include "neighborhood_cut.hpp"
#include "../heuristics/RandomSegmentPacking.hpp"

struct ReplacementBucket{
    std::vector<segment> replacements;
    cost_t gain;

    std::vector<segment>::iterator begin(){ return  replacements.begin(); }
    std::vector<segment>::iterator end(){ return  replacements.end(); }

    void push(segment& s){ replacements.push_back(s); }
    [[nodiscard]] int size() const{ return static_cast<int>(replacements.size()); }
};

class neighborhood_gtsp: public neighborhood{
public:
    neighborhood_gtsp(cppied_context& c, cppied_instance& i, cppied_solution& s) :
        neighborhood(c,i,s), tspEngine(ctx), trial(){}

    bool local_search() override;

    std::array<segment, 3> segment_to_nodes(const segment& s);
    cost_t gain(const cppied_solution &pSol, sVecIt s, int M);
protected:
    tsp_engine tspEngine;

    void reset();

    void setReplacements();
    std::vector<int> selectClusters();
    std::vector<segment> getReplacements(std::vector<int>& unsatisfiedCells);
    std::vector<int> getUnsatisfiedCells();
    ReplacementBucket getReplacementBucket(int index);
    void filterReplacementBucket(ReplacementBucket& bucket,
                                 std::vector<int>& unsatisfiedCells);

    tsp::GTSPGraph buildGtspInstance(std::vector<int>& selection);
    void buildDummyClusters(tsp::GTSPGraph& emptyInstance);
    void addSegmentClusters(tsp::GTSPGraph& instance);
    void addSegmentToClusters(tsp::GTSPGraph& instance, segment& s);
    void addReplacementsToClusters(tsp::GTSPGraph& instance, int index);
    void addClustersNodesToTour(tsp::GTSPGraph& instance);
    void setGtspCosts(tsp::GTSPGraph& instance);
    void setRootCosts(tsp::GTSPGraph& instance);
    void setNodeSegmentCosts(tsp::GTSPGraph& instance, int node);
    void setForwardNodesCosts(tsp::GTSPGraph& instance, int node);
    void transformTspTourToSolution(tsp::GTSPGraph& instance);
    void orderTspTour(tsp::GTSPGraph& instance);
    int getClusterGroupSize(tsp::GTSPGraph& instance, int node);
    void addClusterGroupSegmentToTrial(tsp::GTSPGraph& instance, int node);

private:
    cppied_solution trial;
    std::vector<ReplacementBucket> replacements;
};