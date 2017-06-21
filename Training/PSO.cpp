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

    particle->network->SetWeights(new_position);
    new_error       = particle->network->GetMeanSquaredError(*particle->train_data);
    particle->error = new_error;

    //Compare current error with best particle error
    if (new_error < particle->best_error)
    {
        particle->best_error       = new_error;
        particle->best_position    = new_position;
    }

    pthread_exit(NULL);
}

std::vector<double> NN::PSO(
    std::vector<std::vector<double> >   train_data,
    uint32_t                            repeat,
    std::vector<Base*>                  base_networks
){
    if (repeat < 1)
        throw std::runtime_error("ERROR [PSO]: Repeat must be greater than 0.");
    if (base_networks.size() == 0)
        throw std::runtime_error("ERROR [PSO]: Particle size (base_networks.size()) is zero.");
    for (uint32_t i = 0; i < base_networks.size(); i++)
    {
        if (base_networks[i] == NULL)
            throw std::runtime_error("ERROR [PSO]: Base network" + std::to_string(i) + " is NULL.");
        if (base_networks[0]->GetWeights() != base_networks[i]->GetWeights())
            throw std::runtime_error("ERROR [PSO]: Base networks do not match. { 0 != " + std::to_string(i) + "}");
    }
    if (train_data.size() == 0 || train_data[0].size() == 0)
        throw std::runtime_error("ERROR [PSO]: Train data is equal to zero.");
    
    double                      MIN                         = -10.0;
    double                      MAX                         = +10.0;
    uint32_t                    particles_count             = base_networks.size();
    uint32_t                    weights_length              = base_networks[0]->GetWeights().size();
    uint32_t                    repeat_counter              = 0;
    double*                     best_global_error           = new double(std::numeric_limits<double>::max());
    std::vector<double>*        best_global_position        = new std::vector<double>(weights_length);
    std::vector<Particle>*      swarm                       = new std::vector<Particle>(particles_count);

    //Setup Swarm
    for (uint32_t i = 0; i < particles_count; i++)
    {
        //Compute Velocity
        std::vector<double> velocity(weights_length);
        double low      = 0.1 * MIN;
        double high     = 0.1 * MAX;
        for (uint32_t j = 0; j < weights_length; j++)
        {
            velocity[j] = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
        }
        //Compute Error
        double error = base_networks[i]->GetMeanSquaredError(train_data);
        //Setup Particle
        {
            (*swarm)[i]                 = Particle();
            (*swarm)[i].position        = base_networks[i]->GetWeights();
            (*swarm)[i].velocity        = velocity;
            (*swarm)[i].best_position   = (*swarm)[i].position;
            (*swarm)[i].error           = error;
            (*swarm)[i].best_error      = error;
            (*swarm)[i].thread_id       = new pthread_t();
            (*swarm)[i].network         = base_networks[i];

            //Globals
            (*swarm)[i].best_global_position   = best_global_position;
            (*swarm)[i].best_global_error      = best_global_error;

            //Train Data
            (*swarm)[i].train_data             = &train_data;
        }

        //Update Global Variables
        if (*best_global_error > (*swarm)[i].error)
        {
            *best_global_position    = (*swarm)[i].position;
            *best_global_error       = (*swarm)[i].error;
        }
    }

    //Start Training
    while(repeat_counter < repeat)
    {
        for (uint32_t i = 0; i < particles_count; i++)
        {
            pthread_create((*swarm)[i].thread_id, NULL, &NN::PSOParticleThread, (void*) &(*swarm)[i]);
        }

        for (uint32_t i = 0; i < particles_count; i++)
        {
            pthread_join(*(*swarm)[i].thread_id, NULL);
            //Update Global Variables
            if (*best_global_error > (*swarm)[i].error)
            {
                *best_global_position    = (*swarm)[i].position;
                *best_global_error       = (*swarm)[i].error;
            } 
        }
        std::cout << "Epoch: " << std::to_string(repeat_counter) << std::endl;
        repeat_counter++;
    }

    std::vector<double> rtn = *best_global_position;
    delete best_global_position;
    delete best_global_error;
    delete swarm;
    return rtn;
}