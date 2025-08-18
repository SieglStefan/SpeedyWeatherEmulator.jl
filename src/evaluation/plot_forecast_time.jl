module PlotForecastTime

using SpeedyWeather
using Plots
using Statistics
using ..BasicStructs
using ..GeneralUtils
using ..SimDataHandling

export plot_forecast_time


function plot_forecast_time(sim_para::SimPara;  
                            t_spinup::Real = 10.,    # spinup time (settle-in time)
                            t_step::Real = 6.)

    sim_para_plot = SimPara(sim_para.trunc, sim_para.n_steps, 10)                                                
    sim_data = create_sim_data(sim_para_plot, t_spinup=t_spinup, t_step=t_step)

    vor = abs.(sim_data.data)
    n_vars = length(vec(vor[:,1,1]))  # number of complex spectral coeff.

    n_steps = sim_para.n_steps 
    t_max = t_spinup + (n_steps-1) * t_step

    mean_vor = mean(vor, dims=3)
    
      
    p = Plots.plot(t_spinup:t_step:t_max, mean_vor[1,:,1], labels=false)

    for i in 2:n_vars
        Plots.plot!(t_spinup:t_step:t_max, mean_vor[i,:,1], labels=false)
    end

    return p
end


end