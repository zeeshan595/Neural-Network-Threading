#ifndef _PSO
#define _PSO

namespace NN
{
    struct Particle
    {
        std::vector<double>     velocity;

        std::vector<double>     position;
        std::vector<double>     best_position;

        double                  error               = 0.0;
        double                  best_error          = 0.0;
        
        Base*                   network             = NULL;
        pthread_t*              thread_id           = NULL;

        //Global
        std::vector<double>*    best_global_position;
        double*                 best_global_error;

        //Train Data
        std::vector<std::vector<double> >*  train_data;
    };

    void* PSOParticleThread(void* attr);
    std::vector<double> PSO(
        std::vector<std::vector<double> >   train_data,
        uint32_t                            repeat,
        std::vector<Base*>                  base_networks
    );
};

#include "PSO.cpp"
#endif