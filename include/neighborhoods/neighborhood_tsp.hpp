#pragma once

#include "neighborhood.hpp"
#include "../engines/tsp_engine.hpp"
#include "neighborhood_trim.hpp"
#include "neighborhood_cut.hpp"

struct TspInstance{
    Eigen::MatrixXi distances;
    std::vector<int> tour;
};

class neighborhood_tsp : public neighborhood{
public:
    neighborhood_tsp(cppied_context& c, cppied_instance& i,  cppied_solution& s) :
        neighborhood(c,i,s), tspEngine(ctx), trial(), tspDimension(static_cast<int>(s.path.size())*3 + 3){}

    bool local_search() override;
    segment tour_id_to_segment(int i);
protected:
    TspInstance buildTspInstance();
    Eigen::MatrixXi getDistanceMatrix();

    void connectDummyNodes(Eigen::Ref<Eigen::MatrixXi> distances);
    void connectSegmentsToEndNode(Eigen::Ref<Eigen::MatrixXi> distances) const;
    void connectPairWiseSegments(const Eigen::Ref<Eigen::MatrixXi>& distances);

    void connectSegmentNodes(Eigen::Ref<Eigen::MatrixXi> distances, int path_index);
    void connectSegmentToForwardNodes(const Eigen::Ref<Eigen::MatrixXi>& distances, int path_index);
    void connectSegmentPair(Eigen::Ref<Eigen::MatrixXi> distances, int first, int second);

    void splitTrial();
    std::vector<int> selectSplitIndexes();
    std::vector<segment> constructSplitPath(std::vector<int>& splitSelection);
    int getSplitIndex(const segment& s);
    void splitAndPush(std::vector<segment>& partialPath, const segment& s);
    void moveTrialFromTspSolution(TspInstance& instance);
    void setTrialFromTspTour(std::vector<int>& tour);
    void orderTspTourAndOrientation(std::vector<int>& unorderedTour);

private:
    tsp_engine tspEngine;
    cppied_solution trial;
    int tspDimension;
};