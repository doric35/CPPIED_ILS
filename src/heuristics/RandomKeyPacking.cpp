#include "../../include/heuristics/RandomKeyPacking.hpp"

std::vector<Packing> RandomKeyPacking::solve() {
    std::shuffle(collection.begin(), collection.end(), rng);
    for (auto& k : collection)
        findPacking(k);

    return packing;
}

void RandomKeyPacking::findPacking(const Key &key) {
    for (auto& bin : packing){
        if (!bin.contains(key)) {
            bin.push(key);
            return;
        }
    }
    addNewPacking(key);
}

void RandomKeyPacking::addNewPacking(const Key &key) {
    packing.emplace_back();
    packing.back().push(key);
}