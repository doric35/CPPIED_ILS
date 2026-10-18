#pragma once

#include "../structures.hpp"
#include "../utils.hpp"
#include "../cppied_method_base.hpp"

struct Insertion{
    std::pair<int,int> indexes{-1,-1};
    cost_t cost {MAX_INT, MAX_INT};
    std::pair<segment, segment> segments;
};

inline uint64_t encodeSegment(int a, int b) {
    return (uint64_t(a) << 32) | uint32_t(b);
}

inline void boost_hash_combine(std::size_t &seed, const uint64_t& value){
seed ^= std::hash<uint64_t>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct SegmentPair{
    segment a, b;
    SegmentPair(const segment& first, const segment& second) : a(first), b(second){
        if (b < a) std::swap(a, b);
    }
    auto operator<=>(const SegmentPair&) const = default;
};

struct OrderedPicks{
    const std::array<std::pair<int, cost_t>, 4>& picks;
    const std::array<int, 4>& priority;
};

struct HashSegmentPair{
    std::size_t operator()(const SegmentPair& pair) const{
        std::size_t seed = 0;
        boost_hash_combine(seed, encodeSegment(pair.a.source, pair.a.target));
        boost_hash_combine(seed, encodeSegment(pair.b.source, pair.b.target));
        return seed;
    }
};

class DominatedInsertion : public std::runtime_error{
public:
    explicit DominatedInsertion(const std::string& message) : std::runtime_error(message){}
};

class MinimumSegmentPairInsertion {
private:
    cppied_instance& instance;
    coverage_engine& coverage;
    path_engine&     geometry;
    cppied_solution& solution;

    segment firstSegment;
    segment firstReversedSegment;
    segment secondSegment;
    segment secondReversedSegment;

    Insertion incumbent;
    cost_t bound;

    std::array<std::pair<segment, segment>, 8> pairedKernels;

    std::unordered_map<SegmentPair, Insertion, HashSegmentPair> memoizedMinimalInsertions;
    std::unordered_map<std::uint64_t, std::array<std::pair<int,cost_t>, 4>> memoizedDisjointTopInsertions;

    void solveUnitaryInsertion();
    void setIncumbentFromUnitaryInsertion(std::array<std::pair<int, cost_t>, 4>& pick);

    void solveDisjointInsertion();
    std::array<std::pair<int, cost_t>, 4> getSegmentTopInsertions(segment& s);
    void setIncumbentFromDisjointInsertions(std::array<std::pair<int, cost_t>, 4>& firstPick,
                                            std::array<std::pair<int, cost_t>, 4>& secondPick);
    void setIncumbentFromDisjointInsertionPairs(std::array<Insertion, 16> insertionPairs);

    void solvePairedInsertion();
    void pairedInsertionKernel(int index);

    cost_t getInsertionCost(segment &s, int index);
    cost_t getInsertionCost(Insertion& insertion);

    void updateIncumbent(Insertion& trial);
    void correctIncumbentSegmentsOrder();

    bool picksAreDominated(std::array<std::pair<int, cost_t>, 4>& firstPick,
                           std::array<std::pair<int, cost_t>, 4>& secondPick);

protected:
    void solveUnitPathEdgeCase();
    std::array<Insertion, 16> getDisjointInsertionPairs(std::array<std::pair<int, cost_t>, 4>& firstPick,
                                                        std::array<std::pair<int, cost_t>, 4>& secondPick);

public:
    //----Constructors & Destructors----
    MinimumSegmentPairInsertion(cppied_instance& instance,
                                coverage_engine& coverage,
                                path_engine&     geometry,
                                cppied_solution& solution,
                                segment first, segment second,
                                cost_t bound) :
                                instance(instance), coverage(coverage),
                                geometry(geometry), solution(solution),
                                firstSegment(first), firstReversedSegment(geometry.flip_segment(first)),
                                secondSegment(second), secondReversedSegment(geometry.flip_segment(second)),
                                incumbent(), bound(bound), memoizedMinimalInsertions(),
                                memoizedDisjointTopInsertions(), pairedKernels(){
        pairedKernels = {{
             {firstSegment, secondSegment},
             {firstSegment, secondReversedSegment},
             {firstReversedSegment, secondSegment},
             {firstReversedSegment, secondReversedSegment},
             {secondSegment, firstSegment},
             {secondSegment, firstReversedSegment},
             {secondReversedSegment, firstSegment},
             {secondReversedSegment, firstReversedSegment}
        }};
    }

    ~MinimumSegmentPairInsertion() = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    Insertion solve();

    //----Static Functions----

    //----Friend Classes----
};
