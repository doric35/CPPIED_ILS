#pragma once

#include "../structures.hpp"
#include "../cppied_method_base.hpp"

struct Key{
    int i;
    int k;
    int l;
};

struct Packing{
private:
    std::vector<int> segmentIndexes;
    std::set<int> positionsIndexes;

public:
    bool contains(const Key& key){
        bool contains_i = key.i >=0 && positionsIndexes.contains(key.i);
        bool contains_k = key.k >=0 && positionsIndexes.contains(key.k);
        bool contains_l = key.l >= 0 && positionsIndexes.contains(key.l);
        return contains_i || contains_k || contains_l;
    }

    void push(const Key& key){
        segmentIndexes.push_back(key.i);
        for (int offset = -1; offset <= 1; offset++)
            positionsIndexes.insert(key.i+offset), positionsIndexes.insert(key.k+offset), positionsIndexes.insert(key.l+offset);
    }

    std::vector<int>::const_iterator begin(){ return segmentIndexes.begin(); }
    std::vector<int>::const_iterator end(){ return segmentIndexes.end(); }
};

class RandomKeyPacking {
private:
    std::vector<Packing> packing;
    std::vector<Key>& collection;

    std::random_device rnd;
    std::mt19937 rng;
protected:
    void findPacking(const Key& key);
    void addNewPacking(const Key& key);

public:
    //----Constructors & Destructors----
    RandomKeyPacking(std::vector<Key>& collection) : packing(), collection(collection), rnd(), rng(rnd()) {}

    ~RandomKeyPacking() = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    std::vector<Packing> solve();

    //----Static Functions----

    //----Friend Classes----
};
