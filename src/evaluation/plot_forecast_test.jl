using Plots
using Statistics


"""
    plot_forecast_test(sim_data::SimData)  

Plots the spectral coefficients to test the SpeedyWeather.jl forecast quality.

Plots the mean spectral coefficients (of the first 50 initial conditions) in the time interval 
    `[t_spinup, t_max] = [t_spinup, t_spinup + (n_data-1) * t_step]` to test, if the SpeedyWeather.jl simulation
    defined by `sim_data.sim_para` makes sense.
- If the spectral coefficients strongly diverge, `n_data` or `t_step` is probably too large.
- If an initial condition is given, `t_spinup` is ignored. Therefore, the time interval `[0, (n_data-1) * t_step]` is used.


# Arguments
- `sim_data::SimData`: Contains the spectral coefficients plotted (only a part of `sim_data` is plotted).

# Returns
- `p::Plots.Plot`: Plot of the mean spectral coefficents in the interval `[t_spinup, t_max]`.
"""
function plot_forecast_test(sim_data::SimData)
    n_data = sim_data.sim_para.n_data
    t_step = sim_data.sim_para.t_step
    n_spinup = sim_data.sim_para.n_spinup
    trunc = sim_data.sim_para.trunc

    data = sim_data.data[:,:,1:50]
    
    vor = abs.(data)
    mean_vor = mean(vor, dims=3)

    n_coeff = calc_n_coeff(trunc=trunc)           # number of complex spectral coeff.
    t_max = (n_spinup + n_data - 1) * t_step


    # If a initial condition is used, do NOT spinup the system
    if sim_data.sim_para.initial_cond !== nothing
        t_spinup = 0.
    else
        t_spinup = n_spinup * t_step
    end

    # Plotting all coefficients
    p = Plots.plot(t_spinup:t_step:t_max, mean_vor[1,:,1], label=false)

    for i in 2:n_coeff
        Plots.plot!(t_spinup:t_step:t_max, mean_vor[i,:,1], label=false)
    end

    return p
end


