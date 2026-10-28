#pragma once

#include "../utils.hpp"

class MaximumSpaceConstrainedArraySelection {
private:

protected:
    int spacingConstraint;

    const std::vector<double>& costs;
    std::vector<int> selection;
    std::vector<double> values;

    void setStateValues();
    void setInitialStateValues();
    void setRecursiveStateValues();
    void setSelection();

public:
    //----Constructors & Destructors----
    MaximumSpaceConstrainedArraySelection(int spacingConstraint, const std::vector<double>& costs) :
        spacingConstraint(spacingConstraint), costs(costs), selection(), values(costs.size(), 0.0) {}

    ~MaximumSpaceConstrainedArraySelection() = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    const std::vector<int>& solve();

    //----Static Functions----

    //----Friend Classes----
};
