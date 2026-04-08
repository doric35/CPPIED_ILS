#include "../../include/perturbations/perturbation.hpp"

void perturbation::perturbate(cppied_solution &pSol) {
    d_perturbate(pSol);
    geometry.complete(pSol);
    coverage.reset(pSol);
}