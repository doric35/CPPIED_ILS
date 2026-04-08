#pragma once

#include "../utils.hpp"

class luby {
public:
    int operator()(int i);
protected:
    int scale=4;
};