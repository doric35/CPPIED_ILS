#pragma once

#include "utils.hpp"

struct dimension{
    const unsigned W;
    const unsigned H;
};

struct offset{
    const int dx;
    const int dy;
};

struct S{
    unsigned LB;
    unsigned UB;
};

struct B{
    unsigned LB;
    unsigned UB;
};

struct alphabet{
    const std::vector<unsigned> types;
};

struct configuration{
    dimension d;
    std::vector<unsigned> c;
    void swap(configuration& other){
        c.swap(other.c);
    }
};

struct neighborhood{
    dimension d;
    const std::vector<offset> V;
    neighborhood(const std::vector<offset>& pV, const dimension& pD) : V(pV), d(pD) {}
    unsigned* pointer_arithmetic(const offset& o, unsigned* cell) const{
        return cell + (o.dy * d.W) + o.dx;
    }
};

struct moore_neighborhood : neighborhood{
    moore_neighborhood(const dimension& pD) :
    neighborhood({
            {-1, -1}, {0, -1}, {1, -1},
            {-1, 0}, {0, 0}, {1, 0},
            {-1, 1}, {0, 1}, {1, 1}},
             pD
            ){}
};

struct transition{
    std::size_t arity = 1;
    alphabet A;
    virtual unsigned delta(const neighborhood& N, unsigned* c) const = 0;
};

struct identity : transition{
    unsigned delta(const neighborhood&N, unsigned* c) const override{
        return *c;
    }
};

struct game_of_life : transition {
    struct neighborhood_statistics{
        unsigned better=0;
        unsigned lower=0;
        unsigned equals=0;
    };
    neighborhood_statistics get_stats(const neighborhood&N, unsigned* c) const{
        neighborhood_statistics s;
        for (auto& nb : N.V){
            unsigned* c_hat = N.pointer_arithmetic(nb, c);
            s.better += static_cast<unsigned>(*c_hat > *c);
            s.lower += static_cast<unsigned>(*c_hat < *c);
            s.equals += static_cast<unsigned>(*c_hat == *c);
        }
        return s;
    }
    unsigned delta (const neighborhood&N, unsigned* c) const override{
        if (*c <= 0) return *c; //IsDead
        neighborhood_statistics s = get_stats(N, c);
        if (s.better + s.equals < 2)
            return static_cast<unsigned>(std::max(A.types[0], *c - 1));
        else if (s.better + s.equals > 3)
            return static_cast<unsigned>(std::max(A.types[0], *c - 1));
        else if (s.better + s.equals == 3)
            return static_cast<unsigned>(std::min(A.types.back(), *c + 1));
        else
            return *c + 1;

    }
};

struct weighted_life : transition {
    std::vector<unsigned> W;
    unsigned delta (const neighborhood&N, unsigned* c) const override{
        if (W.size() != N.V.size())
            throw std::runtime_error("Encountered a mismatch between weights and neighborhood in wieghted life.\n");

    }
};

struct evolution{
    neighborhood N;
    std::unique_ptr<transition> T;
    void F(configuration& C){
        configuration next_gen_config = C;
        for (int i = 1; i< N.d.H-1; i++){
            for (int j=1; j< N.d.W-1; j++){
                unsigned* cell_ptr = &C.c[i * N.d.W + j];
                next_gen_config.c[i * N.d.W + j] = T->delta(N, cell_ptr);
            }
        }
        C.swap(next_gen_config);
    }
};