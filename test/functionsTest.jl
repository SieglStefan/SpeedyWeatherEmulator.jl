using SpeedyWeatherEmulator
using Test

@test add_two(2.) ≈ 4.

#for i in 1:54
    #println("coeff ", i, " : ", is_coeff_zero(i, sim_data))
#end

sim_para = SimPara(trunc = 5, n_steps=8, n_ic = 10)
sim_data = create_sim_data(sim_para)
data = sim_data.data

display(plot_forecast_time_new(sim_para, t_step=6.0, t_spinup=30.0))

display(data)