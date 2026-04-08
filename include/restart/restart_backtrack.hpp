#pragma once

#include "restarts.hpp"
#include "../construction/dp_sweeper.hpp"

class restart_backtrack : public  restarts{
public:
    using  restarts::restarts;
    void restart(cppied_solution&) override;
    void sample(cppied_solution&);
protected:
    std::queue<std::vector<bool>> history{};
    std::vector<segment> warm_start{};
};