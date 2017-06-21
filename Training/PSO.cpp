std::vector<double> NN::PSO(
    std::vector<std::vector<double> >   train_data,
    uint32_t                            particle_count,
    uint32_t                            repeat,
    Base*                               (*network_creation_func)()
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
    uint32_t                        global_repeat_counter       = 0;
    pthread_cond_t                  global_cond_var             = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t                 sync_mutex                  = PTHREAD_MUTEX_INITIALIZER;
    
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
        swarm[i]->thread_mutex          = PTHREAD_MUTEX_INITIALIZER;
        swarm[i]->sync_mutex            = &sync_mutex;
        swarm[i]->thread_cond_var       = PTHREAD_COND_INITIALIZER;
        swarm[i]->global_cond_var       = &global_cond_var;
        swarm[i]->repeat_counter        = 0;
        swarm[i]->repeat_amount         = repeat;
        swarm[i]->global_repeat_counter = &global_repeat_counter;
    }

    //Create threads
    for (uint32_t i = 0; i < particle_count; i++)
    {
        if (pthread_create(&swarm[i]->thread_id, NULL, &PSOParticleThread, swarm[i]) != 0)
        {
            throw std::runtime_error("ERROR [PSO]: Unable to create thread " + std::to_string(i) + ".");
        }
    }

    //Train Neural Network
    while(global_repeat_counter < repeat)
    {
        pthread_mutex_lock(&sync_mutex);
        //Wait for other threads
        for (uint32_t i = 0; i < particle_count; i++)
        {
            while (swarm[i]->repeat_counter <= global_repeat_counter)
            {
                pthread_cond_wait(&swarm[i]->thread_cond_var, &sync_mutex);
            }
            
        }
        std::cout << "Epoch: " << global_repeat_counter << " MSR: " << best_global_error << std::endl;
        pthread_cond_broadcast(&global_cond_var);
        global_repeat_counter++;
        pthread_mutex_unlock(&sync_mutex);
    }

    //Wait for threads
    for (uint32_t i = 0; i < particle_count; i++)
    {
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
    Particle* particle = (Particle*) attr;
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

    while (particle->repeat_counter < particle->repeat_amount)
    {
        //Wait for other threads
        pthread_mutex_lock(particle->sync_mutex);
        while (particle->repeat_counter > *particle->global_repeat_counter)
        {
            pthread_cond_wait(particle->global_cond_var, particle->sync_mutex);
        }

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
            pthread_mutex_lock(&particle->thread_mutex);
            *particle->best_global_position    = particle->position;
            *particle->best_global_error       = particle->error;
            pthread_mutex_unlock(&particle->thread_mutex);
        }
        pthread_cond_signal(&particle->thread_cond_var);
        particle->repeat_counter++;
        pthread_mutex_unlock(particle->sync_mutex);
    }
    pthread_exit(NULL);
}