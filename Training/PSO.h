#ifndef _PSO
#define _PSO

namespace NN
{
    struct Particle
    {
        //Particle Variables
        std::vector<double>                 velocity;

        std::vector<double>                 position;
        std::vector<double>                 best_position;

        double                              error                       = -1.0;
        double                              best_error                  = -1.0;

        //Network
        Base*                               particle_network            = NULL;

        //Global Variables
        std::vector<std::vector<double> >*  train_data                  = NULL;
        std::vector<double>*                best_global_position        = NULL;
        double*                             best_global_error           = NULL;
    
        //Threading & Sync
        pthread_t                           thread_id;
        pthread_mutex_t                     thread_mutex;
        uint32_t                            repeat_counter              = -1;
        uint32_t                            repeat_amount               = -1;
    };

    void* PSOParticleThread(void* attr);
    std::vector<double> PSO(
        std::vector<std::vector<double> >   train_data,
        uint32_t                            particle_count,
        uint32_t                            repeat,
        Base*                               (*network_creation_func)()
    );
};

#include "PSO.cpp"
#endif