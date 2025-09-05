using SpeedyWeatherEmulator

sim_para = SimPara(trunc=5, n_data=48, n_ic=1000)
generate_raw_data(sim_para)

sim_data = SimData(sim_para)
save_data(sim_data)