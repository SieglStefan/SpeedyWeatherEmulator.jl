using SpeedyWeatherEmulator
using Random


# Fix the seed for reproducibility
Random.seed!(1234)


# Define simulation parameters
const TRUNC = 5
const N_DATA = 48
const N_IC = 1000

sim_para = SimPara(trunc=TRUNC, n_data=N_DATA, n_ic=N_IC)


# Generate raw data
generate_raw_data(sim_para)


# Generate simulation data and save it
sim_data = SimData(sim_para)
save_data(sim_data)