#pragma once

#include "../utils.hpp"
#include "../neighborhoods/neighborhood.hpp"
#include "../neighborhoods/neighborhood_reduce.hpp"
#include "../neighborhoods/neighborhood_cut.hpp"
#include "../neighborhoods/neighborhood_trim.hpp"
#include "../neighborhoods/neighborhood_n1.hpp"
#include "../neighborhoods/neighborhood_n12.hpp"
#include "../neighborhoods/neighborhood_n21.hpp"
#include "../neighborhoods/neighborhood_nested.hpp"
#include "../neighborhoods/neighborhood_tsp.hpp"
#include "../neighborhoods/neighborhood_gtsp.hpp"
#include "../neighborhoods/neighborhood_r.hpp"
#include "../perturbations/perturbation.hpp"
#include "../perturbations/perturbation_ri.hpp"
#include "../perturbations/perturbation_pt.hpp"
#include "../perturbations/perturbation_db.hpp"
#include "../restart/restart_backtrack.hpp"
#include "../restart/Restart.hpp"

enum class PerturbationType : std::uint8_t {
    RANDOM_INSERTION,
    PARALLEL_TRACK,
    DOUBLE_BRIDGE
};

enum class RestartType : std::uint8_t {
    BACKTRACK
};

enum class LocalSearchType : std::uint8_t {
    N1,
    N12,
    N21,
    NN,
    NTSP,
    NGTSP,
    NR
};

template<typename T>
constexpr std::size_t typeBit(T p)
{
    return static_cast<std::size_t>(p);
}

template <std::size_t N>
std::bitset<N> slice(const std::bitset<11>& bits, std::size_t start)
{
    std::bitset<N> result;
    for (std::size_t i = 0; i < N; ++i)
        result[i] = bits[start + i];
    return result;
}

template<typename T>
class Pool{
public:
    using TMakePtr = std::unique_ptr<T>(cppied_context&,
                                        cppied_instance&,
                                        cppied_solution&);
    using TFactory = std::vector<std::function<TMakePtr>>;
    Pool() : Ts(), selection(), rd(), rng(rd()), dist(){}
    Pool(TFactory pTs) : Ts(std::move(pTs)), selection(Ts.size()), rd(), rng(rd()), dist(0, std::max(0,static_cast<int>(Ts.size())-1)){
        std::iota(selection.begin(), selection.end(), 0);
    }

    void push(const std::function<TMakePtr>& item_type){
        Ts.push_back(item_type);
        selection.push_back(selection.size());
        dist = std::uniform_int_distribution<>(0, std::max(0,static_cast<int>(Ts.size())-1));
    }

    std::function<TMakePtr> select(){
        int i = dist(rng);
        return Ts[i];
    }

    [[nodiscard]] bool empty() const{ return Ts.empty(); }
private:
    TFactory Ts;
    std::vector<int> selection;

    std::random_device rd;
    std::uniform_int_distribution<> dist;
    std::mt19937 rng;
};

template<typename T>
class Sequence{
public:
    using TMakePtr = std::unique_ptr<T>(cppied_context&,
                                        cppied_instance&,
                                        cppied_solution&);
    using TFactory = std::vector<std::function<TMakePtr>>;
    Sequence() : Ts(){}
    Sequence(TFactory Ts) : Ts(std::move(Ts)){}

    TFactory::iterator begin(){ return Ts.begin(); }
    TFactory::iterator end(){ return Ts.end(); }

    std::function<TMakePtr> operator[](std::size_t i){
        return Ts[i];
    }

    [[nodiscard]] bool empty() const{ return Ts.empty(); }

    void push(const std::function<TMakePtr>& item_type){
        Ts.push_back(item_type);
    }
    [[nodiscard]] std::size_t size() const { return Ts.size(); }
private:
    TFactory Ts;
};

class LocalSearchFactory{
public:
    using configuration = std::bitset<7>;

    Sequence<neighborhood> setupNeighborhoodsFactory;
    Sequence<neighborhood> simpleNeighborhoodsFactory;
    Sequence<neighborhood> nestedNeighborhoodsFactory;
    Sequence<neighborhood> largeNeighborhoodsFactory;

    LocalSearchFactory(configuration config) :
        setupNeighborhoodsFactory(),
        simpleNeighborhoodsFactory(),
        nestedNeighborhoodsFactory(),
        largeNeighborhoodsFactory() {
        setupNeighborhoodsFactory.push(
                [](cppied_context& c, cppied_instance& i, cppied_solution& s){
                    return std::make_unique<neighborhood_reduce>(c, i, s); });
        setupNeighborhoodsFactory.push(
                [](cppied_context& c, cppied_instance& i, cppied_solution& s){
                    return std::make_unique<neighborhood_trim>(c, i, s); });
        setupNeighborhoodsFactory.push(
                [](cppied_context& c, cppied_instance& i, cppied_solution& s){
                    return std::make_unique<neighborhood_cut>(c, i, s); });

        if (config[typeBit(LocalSearchType::N1)])
            simpleNeighborhoodsFactory.push(
                    [](cppied_context& c, cppied_instance& i, cppied_solution& s){
                        return std::make_unique<neighborhood_n1>(c, i, s); });
        if (config[typeBit(LocalSearchType::N12)])
            simpleNeighborhoodsFactory.push(
                    [](cppied_context& c, cppied_instance& i, cppied_solution& s){
                        return std::make_unique<neighborhood_n12>(c, i, s); });
        if (config[typeBit(LocalSearchType::N21)])
            simpleNeighborhoodsFactory.push(
                    [](cppied_context& c, cppied_instance& i, cppied_solution& s){
                        return std::make_unique<neighborhood_n21>(c, i, s); });
        if (config[typeBit(LocalSearchType::NN)])
            nestedNeighborhoodsFactory.push(
                    [](cppied_context& c, cppied_instance& i, cppied_solution& s){
                        return std::make_unique<neighborhood_nested>(c, i, s); });
        if (config[typeBit(LocalSearchType::NTSP)])
            largeNeighborhoodsFactory.push(
                    [](cppied_context& c, cppied_instance& i, cppied_solution& s){
                        return std::make_unique<neighborhood_tsp>(c, i, s); });
        if (config[typeBit(LocalSearchType::NGTSP)])
            largeNeighborhoodsFactory.push(
                    [](cppied_context& c, cppied_instance& i, cppied_solution& s){
                        return std::make_unique<neighborhood_gtsp>(c, i, s); });
        if (config[typeBit(LocalSearchType::NR)])
            largeNeighborhoodsFactory.push(
                    [](cppied_context& c, cppied_instance& i, cppied_solution& s){
                        return std::make_unique<neighborhood_r>(c, i, s); });
    }
    std::vector<std::unique_ptr<neighborhood>> getSetupNeighborhoods(cppied_context& c, cppied_instance& i, cppied_solution& s){
        std::vector<std::unique_ptr<neighborhood>> nbh;
        for (auto && j : setupNeighborhoodsFactory)
            nbh.push_back(j(c,i,s));
        return nbh;
    }
    std::vector<std::unique_ptr<neighborhood>> getSimpleNeighborhoods(cppied_context& c, cppied_instance& i, cppied_solution& s){
        std::vector<std::unique_ptr<neighborhood>> nbh;
        for (auto && j : simpleNeighborhoodsFactory)
            nbh.push_back(j(c,i,s));
        return nbh;
    }
    std::vector<std::unique_ptr<neighborhood>> getNestedNeighborhoods(cppied_context& c, cppied_instance& i, cppied_solution& s){
        std::vector<std::unique_ptr<neighborhood>> nbh;
        if (!nestedNeighborhoodsFactory.empty())
            nbh.push_back(nestedNeighborhoodsFactory[0](c,i,s));
        return nbh;
    }
    std::vector<std::unique_ptr<neighborhood>> getLargeNeighborhoods(cppied_context& c, cppied_instance& i, cppied_solution& s){
        std::vector<std::unique_ptr<neighborhood>> nbh;
        for (auto && j : largeNeighborhoodsFactory)
            nbh.push_back(j(c,i,s));
        return nbh;
    }
    [[nodiscard]] std::size_t size() const{ return setupNeighborhoodsFactory.size() + simpleNeighborhoodsFactory.size() +
        nestedNeighborhoodsFactory.size() + largeNeighborhoodsFactory.size(); }
protected:

private:
};

class PerturbationFactory{
public:
    using configuration = std::bitset<3>;

    Pool<perturbation> perturbationFactory;

    PerturbationFactory(configuration config) : perturbationFactory() {
        if (config[typeBit(PerturbationType::RANDOM_INSERTION)])
            perturbationFactory.push([](cppied_context& c, cppied_instance& i, cppied_solution& s){
                return std::make_unique<perturbation_ri>(c, i, s); });

        if (config[typeBit(PerturbationType::PARALLEL_TRACK)])
            perturbationFactory.push([](cppied_context& c, cppied_instance& i, cppied_solution& s){
                return std::make_unique<perturbation_pt>(c, i, s); });

        if (config[typeBit(PerturbationType::DOUBLE_BRIDGE)])
            perturbationFactory.push([](cppied_context& c, cppied_instance& i, cppied_solution& s){
                return std::make_unique<perturbation_db>(c, i, s); });

    }
    bool empty() const{ return perturbationFactory.empty(); }

    using TMakePtr = std::unique_ptr<perturbation>(cppied_context&,
                                        cppied_instance&,
                                        cppied_solution&);
    std::function<TMakePtr> sample(){ return perturbationFactory.select(); }
protected:

private:
};

class RestartFactory{
public:
    using configuration = std::bitset<1>;

    Pool<Restart> restartFactory;

    RestartFactory(configuration config) : restartFactory() {
        if (config[typeBit(RestartType::BACKTRACK)])
            restartFactory.push([](cppied_context& c, cppied_instance& i, cppied_solution& s){
                return std::make_unique<restart_backtrack>(c, i, s); });
    }

    bool empty() const{ return restartFactory.empty(); }
    using TMakePtr = std::unique_ptr<Restart>(cppied_context&,
                                                   cppied_instance&,
                                                   cppied_solution&);
    std::function<TMakePtr> sample(){ return restartFactory.select(); }
protected:

private:
};

class HeuristicFactory {
public:
    using configuration = std::bitset<11>;

    template<typename T>
    using TFactory = std::vector<std::unique_ptr<T>()>;

    //----Constructors & Destructors----
    HeuristicFactory(configuration config) :
        localSearchFactory(slice<7>(config, 0)),
        perturbationFactory(slice<3>(config, 7)),
        restartFactory(slice<1>(config, 10)){
    }
    virtual ~HeuristicFactory() = default;

    //----Getters----
    bool isConstructOnly(){
        return restartFactory.empty() && perturbationFactory.empty() && localSearchFactory.size() <=3;
    }
    bool isLocalSearchOnly() {
        return restartFactory.empty() && perturbationFactory.empty();
    }
    bool isWithoutPerturbation(){
        return perturbationFactory.empty();
    }
    bool isWithoutRestart(){
        return restartFactory.empty();
    }
    using PerturbationMakePtr = std::unique_ptr<perturbation>(cppied_context&,
                                        cppied_instance&,
                                        cppied_solution&);
    std::function<PerturbationMakePtr> samplePerturbationMethod(){
        return perturbationFactory.sample();
    }
    using RestartMakePtr = std::unique_ptr<Restart>(cppied_context&,
                                                              cppied_instance&,
                                                              cppied_solution&);
    std::function<RestartMakePtr> sampleRestartMethod(){
        return restartFactory.sample();
    }

    //----Setters----

    //----Member Functions----

    //----Static Functions----

    //----Friend Classes----
private:
    LocalSearchFactory localSearchFactory;
    PerturbationFactory perturbationFactory;
    RestartFactory restartFactory;
protected:

};
