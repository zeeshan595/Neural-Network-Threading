std::vector<double> NN::PSO(
    std::vector<std::vector<double> >   train_data,
    uint32_t                            particle_count,
    uint32_t                            repeat,
    Base*                               (*network_creation_func)(),
    void                                (*training_call_back)(double*, double*, std::vector<double>*)
){
    if (repeat < 1)
        throw std::runtime_error("ERROR [PSO]: Repeat must be greater than 0.");
    if (particle_count < 1)
        throw std::runtime_error("ERROR [PSO]: Particle count must be greater than 0.");
    if (network_creation_func == NULL)
        throw std::runtime_error("ERROR [PSO]: Network creation function is NULL.");
    if (train_data.size() == 0 || train_data[0].size() == 0)
        throw std::runtime_error("ERROR [PSO]: Train data length is equal to zero.");

    //Initialize variables for training
    Base*                           temporary_network           = network_creation_func();
    double                          weights_length              = temporary_network->GetWeights().size();
    std::vector<Particle*>          swarm                       = std::vector<Particle*>(particle_count);
    std::vector<double>             best_global_position        = std::vector<double>(weights_length);
    double                          best_global_error           = std::numeric_limits<double>::max();
    pthread_mutex_t                 global_mutex                = PTHREAD_MUTEX_INITIALIZER;
    
    std::srand(std::time(NULL));

    //Setup each particle
    for (uint32_t i = 0; i < particle_count; i++)
    {
        //Create Particle
        swarm[i]                        = new Particle();

        //Network
        swarm[i]->particle_network      = network_creation_func();

        //Particle
        swarm[i]->error                 = swarm[i]->particle_network->GetMeanSquaredError(train_data);
        swarm[i]->best_error            = swarm[i]->error;
        swarm[i]->position              = swarm[i]->particle_network->GetWeights();
        swarm[i]->best_position         = swarm[i]->position;

        double high = +1.0;
        double low  = -1.0;
        swarm[i]->velocity.resize(weights_length);
        for (uint32_t j = 0; j < weights_length; j++)
        {
            swarm[i]->velocity[j]       = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
        }

        //Global
        swarm[i]->best_global_position  = &best_global_position;
        swarm[i]->best_global_error     = &best_global_error;
        swarm[i]->train_data            = &train_data;

        //Threading
        swarm[i]->thread_id             = pthread_t();
        swarm[i]->global_mutex          = &global_mutex;
        swarm[i]->repeat_amount         = &repeat;
        swarm[i]->repeat_counter        = 0;
    }

    //Create threads
    for (uint32_t i = 0; i < particle_count; i++)
    {
        if (pthread_create(&swarm[i]->thread_id, NULL, &PSOParticleThread, swarm[i]) != 0)
        {
            throw std::runtime_error("ERROR [PSO]: Unable to create thread " + std::to_string(i) + ".");
        }
    }

    uint32_t    threads_done    = 0;
    double      progress        = 0.0;
    while (threads_done != particle_count)
    {
        threads_done    = 0;
        progress        = 0;
        for (uint32_t i = 0; i < particle_count; i++)
        {
            if (repeat == swarm[i]->repeat_counter)
                threads_done++;
            
            progress += (100.0 * (double)swarm[i]->repeat_counter) / ((double)repeat * (double)particle_count);
        }
        if (training_call_back != NULL)
            training_call_back(&progress, &best_global_error, &best_global_position);
    }

    //Wait for threads
    for (uint32_t i = 0; i < particle_count; i++)
    {
        //std::cout << "Waiting On Thread: " << i << std::endl;
        if (pthread_join(swarm[i]->thread_id, NULL) != 0)
        {
            throw std::runtime_error("ERROR [PSO]: Unable to join thread " + std::to_string(i) + ".");
        }
    }

    //Clean up
    delete temporary_network;
    for (uint32_t i = 0; i < particle_count; i++)
    {
        delete swarm[i]->particle_network;
        delete swarm[i];
    }
    return best_global_position;
}

void* NN::PSOParticleThread(void* attr)
{
    Particle*       particle                = (Particle*) attr;
    double          MIN                     = -10.0;
    double          MAX                     = +10.0;
    double          inertia_weight          = 0.729;
    double          cognitive_weight        = 1.49445;
    double          social_weight           = 1.49445;
    double          r1                      = 0;
    double          r2                      = 0;

    std::srand(std::time(NULL));

    uint32_t                weights_length  = particle->velocity.size();
    std::vector<double>     new_velocity(weights_length);
    std::vector<double>     new_position(weights_length);
    double                  new_error;

    while (particle->repeat_counter < *particle->repeat_amount)
    {
        //Update Particle Velocity
        for (uint32_t i = 0; i < weights_length; i++)
        {
            r1 = (double)std::rand() / (double)RAND_MAX;
            r2 = (double)std::rand() / (double)RAND_MAX;
            new_velocity[i] =   ((inertia_weight  	*       particle->velocity[i]) +
                                 (cognitive_weight 	* r1 * (particle->best_position[i]              - particle->position[i])) +
                                 (social_weight 	* r2 * ((*particle->best_global_position)[i] 	- particle->position[i])));
        }
        particle->velocity = new_velocity;

        for (uint32_t i = 0; i < weights_length; i++)
        {
            new_position[i] = particle->position[i] + new_velocity[i];
            //Make sure particle does not go out of bounds.
            //using MIN and MAX variables
            if (new_position[i] < MIN)
                new_position[i] = MIN;
            else if (new_position[i] > MAX)
                new_position[i] = MAX;
        }
        particle->position = new_position;

        particle->particle_network->SetWeights(new_position);
        new_error       = particle->particle_network->GetMeanSquaredError(*particle->train_data);
        particle->error = new_error;

        //Compare current error with best particle error
        if (new_error < particle->best_error)
        {
            particle->best_error       = new_error;
            particle->best_position    = new_position;
        }

        //Update Global Variables
        if (*particle->best_global_error > particle->error)
        {
            if (pthread_mutex_lock(particle->global_mutex) != 0)
            {
                throw std::runtime_error("ERROR [PSOParticleThread]: Mutex lock failure");
            }
            *particle->best_global_position    = particle->position;
            *particle->best_global_error       = particle->error;
            if (pthread_mutex_unlock(particle->global_mutex) != 0)
            {
                throw std::runtime_error("ERROR [PSOParticleThread]: Mutex unlock failure");
            }
        }
        particle->repeat_counter++;
    }
    pthread_exit(NULL);
}