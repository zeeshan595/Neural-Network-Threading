#ifdef __MINGW32__
#include <windows.h>
#endif

//GENERAL
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <ctime>
#include <math.h>
#include <chrono>
#include <iomanip>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>

//DataSets
#include "DataSets/Iris.h"

//Core
#include "Core/Activation.h"
#include "Core/CoreFunctions.h"
#include "Core/Synapse.h"
#include "Core/Neuron.h"

//Structures
#include "Structures/Base.h"
#include "Structures/MFNN.h"

//Training
#include "Training/PSO.h"