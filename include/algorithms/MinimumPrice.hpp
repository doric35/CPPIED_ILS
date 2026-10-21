#pragma once

#include "MaximumMove.hpp"

struct Reallocation{
    segment S;
    double value;
    int R;
    cost_t gain;
};

struct ReallocationDynamicProgram{
    std::vector<double> values;
    std::vector<int> predecessors;
};

class MinimumPrice : public MaximumMove<Reallocation> {
private:

protected:
    const Eigen::Ref<const Eigen::VectorXd> vertexPrices;
    double reallocationPrice;
    double lengthPrice;

    bool isValidMove(Reallocation& move) override { return true; };
    cost_t getGain(Reallocation& move) override;

    void solveHorizontal();
    void solveVertical();
    Reallocation solveEdge(int u, int v);

    void updateIncumbent(Reallocation candidate);

    ReallocationDynamicProgram getForwardDp(int u, int v);
    ReallocationDynamicProgram getBackwardDp(int u, int v);
    Reallocation getReallocation(const ReallocationDynamicProgram& dp);

    virtual double sourceLinkPrice(const segment& s) = 0;
    virtual double internLinkPrice(const segment& s) = 0;
    virtual double targetLinkPrice(const segment& s) = 0;


public:
    //----Constructors & Destructors----
    template<typename... Args>
    MinimumPrice(const Eigen::Ref<const Eigen::VectorXd>& vertexPrices,
                 double reallocationPrice,
                 double lengthPrice,
                 Args&&... args) :
        MaximumMove<Reallocation>(std::forward<Args>(args)...),
        vertexPrices(vertexPrices),
        reallocationPrice(reallocationPrice),
        lengthPrice(lengthPrice){
            incumbent.value = MAX_INT;
        }

    ~MinimumPrice() = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    Reallocation solve() override;

    //----Static Functions----

    //----Friend Classes----
};

class FirstStageMinimumPrice : public MinimumPrice {
private:
protected:
    double sourceLinkPrice(const segment& s) override{
        return geometry.dist(solution.path[index -1], s).length - vertexPrices(s.source);
    }
    double internLinkPrice(const segment& s) override {
        return 1.0 - vertexPrices(s.source);
    }
    double targetLinkPrice(const segment& s) override {
        double c1 = 0.0;
        if (index < solution.path.size() - 1)
            c1 = geometry.dist(s, solution.path[index + 1]).length - geometry.dist(solution.path[index - 1], solution.path[index + 1]).length;
        return c1 - reallocationPrice;
    }
public:
    using MinimumPrice::MinimumPrice;
};

class SecondStageMinimumPrice : public MinimumPrice {
private:
protected:
    double sourceLinkPrice(const segment& s) override{
        cost_t cost = geometry.dist(solution.path[index -1], s);
        return cost.turns - vertexPrices(s.source) - lengthPrice * cost.length;
    }
    double internLinkPrice(const segment& s) override {
        return - vertexPrices(s.source) - lengthPrice;
    }
    double targetLinkPrice(const segment& s) override {
        cost_t cost{0,0};
        if (index < solution.path.size() - 1)
            cost = geometry.dist(s, solution.path[index + 1]) - geometry.dist(solution.path[index - 1], solution.path[index + 1]);
        return cost.turns - reallocationPrice - lengthPrice * cost.length;
    }
public:
    using MinimumPrice::MinimumPrice;
};
