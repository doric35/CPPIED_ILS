#include "../../include/perturbations/perturbation.hpp"

void perturbation::perturbate() {
    d_perturbate();
    geometry.complete(solution);
    coverage.reset(solution);
}