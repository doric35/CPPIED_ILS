#pragma once

#include "../utils.hpp"

struct SubArray{
    int start;
    int end;
    double value;

    bool operator<(const SubArray& rhs) const{ return value < rhs.value; }
};

class MaximumSubarrayKadane {
private:

protected:
    const std::vector<double>& costs;

    SubArray incumbent;
    SubArray current;
public:
    //----Constructors & Destructors----
    MaximumSubarrayKadane(const std::vector<double>& costs) :
        costs(costs), incumbent(), current() {}

    ~MaximumSubarrayKadane() = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    SubArray solve();

    //----Static Functions----

    //----Friend Classes----
};
