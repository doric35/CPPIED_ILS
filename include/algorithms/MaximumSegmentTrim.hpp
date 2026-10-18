#pragma once

#include "MaximumReplacement.hpp"

class MaximumSegmentTrim : public MaximumReplacement{
private:
protected:
    bool isValidMove(ReplacementMove& move) override;
    cost_t getGain(ReplacementMove& move) override;

public:
    using MaximumReplacement::MaximumReplacement;
    ~MaximumSegmentTrim() = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    ReplacementMove solve() override;

    //----Static Functions----

    //----Friend Classes----
};
