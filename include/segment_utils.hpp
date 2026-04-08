#pragma once

constexpr int NULL_NODE = -1;
constexpr int REVERSED_NULL_NODE = -2;

inline bool is_node(int n){
    return n != NULL_NODE && n != REVERSED_NULL_NODE;
}
