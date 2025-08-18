module PlotForecastTest

using SpeedyWeather
using Plots
using Statistics
using ..BasicStructs
using ..SimDataHandling


export plot_forecast_test


"""
    plot_forecast_test(sim_para::SimPara)  

Plots the spectral coefficients to test the SpeedyWeather.jl forecast quaility.

Plots the mean spectral coefficients (of 10 random initial conditions) in the time interval 
    [t_spinup, t_max] = [t_spinup, t_spinup + (n_steps-1) * t_step] to test, if the SpeedyWeahter.jl simulation
    defined by `sim_para` makes sense.
    If the spectral coefficients strongly diverge, n_steps is probably too large.
IMPORTANT: If an initial condition is given, t_spinup is ignored. Therefore, the time interval [0, (n_steps-1) * t_step] is used.

# Arguments
- `sim_para::SimPara`:  Contains the parameters of the tested simulation.

# Returns
- `p`:                  Plots of the mean spectral coefficents in the interval [t_spinup, t_max].
"""
function plot_forecast_test(sim_para::SimPara)
    n_steps = sim_para.n_steps
    t_step = sim_para.t_step
    t_spinup = sim_para.t_spinup


    # Defining test simulation data
    sim_para_plot = SimPara(trunc=sim_para.trunc, 
                            n_steps=n_steps, 
                            n_ic=10,
                            t_spinup = t_spinup,
                            t_step = t_step,
                            initial_cond = sim_para.initial_cond) 

    sim_data = SimData(sim_para_plot)
    
    vor = abs.(sim_data.data)
    mean_vor = mean(vor, dims=3)

    n_coeff = length(vec(vor[:,1,1]))            # number of complex spectral coeff.
    t_max = t_spinup + (n_steps-1) * t_step


    # If a initial condition is used, do NOT spinup the system
    if sim_para.initial_cond !== nothing
        t_spinup = 0.
    end


    p = Plots.plot(t_spinup:t_step:t_max, mean_vor[1,:,1], labels=false)

    for i in 2:n_coeff
        Plots.plot!(t_spinup:t_step:t_max, mean_vor[i,:,1], labels=false)
    end

    return p
end



end