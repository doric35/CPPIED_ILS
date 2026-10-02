#pragma once

#include "../structures.hpp"

namespace cppied_intervals {
    struct Interval {
        cost_t gain;
        int    x1;
        int    x2;
        int    y;
        bool   rotated;
        int    id;

        static bool cmpIntervalY (const Interval& a, int y){ return a.y < y; };
        static bool cmpIntervalsPosition(const Interval& a, const Interval& b){
            if (a.y == b.y)
                return a.x2 < b.x2;
            return a.y < b.y;
        };
    };
    class GroupedIntervalSets;
    class IntervalSet {
    private:
        std::vector<Interval> intervals;
        cost_t                value;

        void sortByPosition(){
            std::sort(intervals.begin(), intervals.end(), Interval::cmpIntervalsPosition);
        }
    protected:
    public:
        IntervalSet() : intervals(), value({0, 0}) {}

        ~IntervalSet() = default;

        using const_iterator = std::vector<Interval>::const_iterator;
        using iterator = std::vector<Interval>::iterator;
        using value_type = Interval;

        [[nodiscard]] const_iterator begin() const { return intervals.begin(); }

        [[nodiscard]] const_iterator end() const { return intervals.end(); }

        void push(const Interval &interval) {
            intervals.push_back(interval);
            value += interval.gain;
        }
        void push(const IntervalSet &other) {
            intervals.insert(intervals.end(), other.intervals.begin(), other.intervals.end());
            value += other.value;
        }

        void push_back(const value_type &interval) {
            intervals.push_back(interval);
            value += interval.gain;
        }

        void push_back(value_type &&interval) {
            intervals.push_back(interval);
            value += interval.gain;
        }

        void push_back(IntervalSet::const_iterator begin, IntervalSet::const_iterator end){
            for (auto it = begin; it != end; it++) {
                intervals.push_back(*it);
                value += it->gain;
            }
        }

        void clear() {
            intervals.clear();
            value = cost_t{0, 0};
        }

        template<typename Pred>
        void erase_if(Pred f) {
            std::erase_if(intervals, f);
            setValue();
        }

        void setValue() {
            value = std::accumulate(intervals.begin(), intervals.end(),
                                    cost_t{0, 0}, [](cost_t c, const Interval &interval) {
                        return c + interval.gain;
                    });
        }

        [[nodiscard]] cost_t getValue() const { return value; }

        [[nodiscard]] bool empty() const { return intervals.empty(); }

        void reserve(std::size_t size) { intervals.reserve(size); }

        const Interval &operator[](std::size_t i) const { return intervals[i]; }

        [[nodiscard]] const Interval &front() const { return intervals.front(); }

        [[nodiscard]] const Interval &back() const { return intervals.back(); }

        [[nodiscard]] std::size_t size() const { return intervals.size(); }

        GroupedIntervalSets groupByRows();
    };

    class GroupedIntervalSets {
    private:
        std::vector<IntervalSet> intervalGroups;
        cost_t                   value;

    protected:

    public:
        GroupedIntervalSets() : intervalGroups(), value({0, 0}) {}

        explicit GroupedIntervalSets(std::size_t size) : intervalGroups(size), value({0, 0}) {}

        ~GroupedIntervalSets() = default;

        [[nodiscard]] std::vector<IntervalSet>::const_iterator begin() const{ return intervalGroups.begin(); }

        [[nodiscard]] std::vector<IntervalSet>::const_iterator end() const{ return intervalGroups.end(); }

        void push(const IntervalSet &intervals) {
            intervalGroups.push_back(intervals);
            value += intervals.getValue();
        }

        void push(const GroupedIntervalSets &intervals) {
            intervalGroups.insert(intervalGroups.end(), intervals.begin(), intervals.end());
            value += intervals.getValue();
        }

        void push_back(IntervalSet::const_iterator begin, IntervalSet::const_iterator end){
            intervalGroups.emplace_back();
            intervalGroups.back().push_back(begin, end);
            value += intervalGroups.back().getValue();
        }

        void pop() {
            value -= intervalGroups.back().getValue();
            intervalGroups.pop_back();
        }

        void clear() {
            intervalGroups.clear();
            value = cost_t{0, 0};
        }

        template<typename Pred>
        void erase_if(Pred f) {
            std::erase_if(intervalGroups, f);
            setValue();
        }

        void setValue() {
            value = std::accumulate(intervalGroups.begin(), intervalGroups.end(),
                                    cost_t{0, 0}, [](cost_t c, const IntervalSet &intervals) {
                        return c + intervals.getValue();
                    });
        }

        [[nodiscard]] bool empty() const { return intervalGroups.empty(); }

        void reserve(std::size_t size) { intervalGroups.reserve(size); }

        [[nodiscard]] std::size_t size() const { return intervalGroups.size(); }

        [[nodiscard]] const IntervalSet &back() const { return intervalGroups.back(); }

        const IntervalSet &operator[](std::size_t i) const { return intervalGroups[i]; }

        std::vector<cost_t> getGroupWiseValues() {
            std::vector<cost_t> values(size());
            for (int            i = 0; i < size(); i++)
                values[i] = intervalGroups[i].getValue();
            return values;
        }

        [[nodiscard]] cost_t getValue() const{ return value; }

        std::vector<GroupedIntervalSets> groupSetsByRows(){
            std::vector<GroupedIntervalSets> grouped;
            for (auto& intervals : intervalGroups)
                grouped.push_back(intervals.groupByRows());

            return grouped;
        }
    };

    inline GroupedIntervalSets IntervalSet::groupByRows() {
        if (intervals.empty()) return {};
        sortByPosition();
        int min_y = intervals.front().y;
        int max_y = intervals.back().y;
        GroupedIntervalSets rows;
        rows.reserve(max_y - min_y + 1);
        auto source = intervals.cbegin();
        for (int y = min_y; y<= max_y; ++y){
            auto end = std::lower_bound(source, intervals.cend(), y + 1, Interval::cmpIntervalY);
            rows.push_back(source, end);
            source = end;
        }
        return rows;
    }

    class Bucket{
    private:
        std::vector<int> elements;
        cost_t value;
        void pushIntervalSet(const IntervalSet& intervals){
            for (auto& interval : intervals)
                elements.push_back(interval.id);
            value += intervals.getValue();
        }
    protected:
    public:
        Bucket() : elements(), value({0,0}){

        }
        Bucket(const IntervalSet& intervals) : elements(), value({0,0}){
            elements.reserve(intervals.size());
            pushIntervalSet(intervals);
        }
        Bucket(const GroupedIntervalSets& groups) : elements(), value({0,0}){
            for (auto& intervals : groups)
                pushIntervalSet(intervals);
        }
        ~Bucket() = default;
    };
}
