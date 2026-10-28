#include "../../include/algorithms/MaximumSpaceConstrainedArraySelection.hpp"

const std::vector<int> &MaximumSpaceConstrainedArraySelection::solve() {
    selection.clear();
    if (!costs.empty()) {
        setStateValues();
        setSelection();
    }
    return selection;
}

void MaximumSpaceConstrainedArraySelection::setStateValues() {
    setInitialStateValues();
    setRecursiveStateValues();
}

void MaximumSpaceConstrainedArraySelection::setSelection() {
    int i;
    for (i = static_cast<int>(costs.size())-1; i>0;){
        if (values[i-1] < values[i]) {
            selection.push_back(i);
            i -= spacingConstraint;
        } else i -=1;
    }
    if (i == 0 && costs[i] > 0.0) selection.push_back(0);
    std::reverse(selection.begin(), selection.end());
}

void MaximumSpaceConstrainedArraySelection::setInitialStateValues() {
    values[0] = std::max(values[0], costs[0]);
    int last = std::min(spacingConstraint, static_cast<int>(costs.size()));
    for (int i=1; i<last; ++i)
        values[i] = std::max(values[i-1], costs[i]);
}

void MaximumSpaceConstrainedArraySelection::setRecursiveStateValues() {
    for (int i=spacingConstraint; i< costs.size(); ++i)
        values[i] = std::max(values[i-1], values[i - spacingConstraint] + costs[i]);
}