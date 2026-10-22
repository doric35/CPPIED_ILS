#pragma once

#include "../structures.hpp"
#include "../utils.hpp"
#include "../cppied_method_base.hpp"
#include "MinimumTopTwoInsertion.hpp"

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
protected:
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
    std::unordered_map<std::uint64_t, TopTwoInsertion> memoizedDisjointTopInsertions;

    void solveDisjointInsertion();
    void setIncumbentFromDisjointInsertions(TopTwoInsertion& firstPick,
                                            TopTwoInsertion& secondPick);

    void solvePairedInsertion();
    void pairedInsertionKernel(int index);

    cost_t getInsertionCost(Insertion& insertion);

    void updateIncumbent(Insertion& trial);
    void correctIncumbentSegmentsOrder();

    bool picksAreDominated(TopTwoInsertion& firstPick, TopTwoInsertion& secondPick);
    void solveUnitPathEdgeCase();

public:
    //----Constructors & Destructors----
    MinimumSegmentPairInsertion(cppied_instance& instance,
                                coverage_engine& coverage,
                                path_engine&     geometry,
                                cppied_solution& solution,
                                cost_t bound) :
                                instance(instance), coverage(coverage),
                                geometry(geometry), solution(solution),
                                firstSegment(), firstReversedSegment(),
                                secondSegment(), secondReversedSegment(),
                                incumbent(), bound(bound), memoizedMinimalInsertions(),
                                memoizedDisjointTopInsertions(), pairedKernels(){
    }

    ~MinimumSegmentPairInsertion() = default;

    //----Getters----

    //----Setters----
    void setSegmentPair(segment& first, segment& second){
        firstSegment = first;
        secondSegment = second;
        firstReversedSegment = geometry.flip_segment(firstSegment);
        secondReversedSegment = geometry.flip_segment(secondSegment);
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
        incumbent = Insertion();
    }
    void setBound(cost_t& nextBound){ bound = nextBound; }

    //----Member Functions----
    Insertion solve();

    //----Static Functions----

    //----Friend Classes----
};
