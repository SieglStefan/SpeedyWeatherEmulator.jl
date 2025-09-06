using SpeedyWeatherEmulator
using Plots
using ProgressMeter
using Measures


### This Code is very similar to the counterpart "hyperpara_opt_coarse_eval.jl":
#       - Parts where the code differs, are marked with comments
#       - Defining a separate function for the hyperparameter optimization is not 
#               necessary for this simple consideration
#       - Moreover, keeping the function general (e.g. number of hyperparameters, 
#               different plot symbols, different forecast lengths) would be too 
#               complicated for this simple consideration.


const TRUNC = 5
const N_DATA = 48
const N_IC = 1000

sim_para_loading = SimPara(trunc=TRUNC, n_data=N_DATA, n_ic=N_IC)
sim_data = load_data(sim_para_loading, type="sim_data")
fd = FormattedData(sim_data)


# Different parameters for the fine optimization
L_list = [1]                                                        # number of hidden layers
W_list_fine = [256, 384, 512, 640, 768, 896, 1024, 1536, 2048]      # number of neurons per hidden layer                           


d = 2*calc_n_coeff(trunc=TRUNC)                                
n_params(d, W, L) = (d*W + W) + (L-1)*(W*W + W) + (W*d + d)
data = Dict{Tuple{Int,Int}, NamedTuple}()                     


@showprogress for L in L_list, W in W_list_fine

    id = "_hyperpara_L$(L)_W$(W)"
    sim_para = SimPara(trunc=TRUNC, n_data=N_DATA, n_ic=N_IC, id_key=id)

    em = load_data(sim_para, type="emulator")
    losses = load_data(sim_para, type="losses")

    err_vec = zeros(N_DATA)

    for steps in 1:N_DATA
        err_vec[steps] = compare_emulator(  em,
                                            x_test=fd.data_pairs.x_valid,
                                            y_test=fd.data_pairs.y_valid,
                                            n_it=steps)

    end

    data[(L,W)] = ( err = err_vec,
                    training_time = losses.training_time,
                    params = n_params(d, W, L))
end


params = [data[(L,W)].params for L in L_list for W in W_list_fine]
times  = [data[(L,W)].training_time for L in L_list for W in W_list_fine]

colors = [L for L in L_list for W in W_list_fine]

scatter(params, times;
        group=colors,
        xlabel="Number of parameters",
        ylabel="Training time [s]",
        title="Training time vs. #params (colored by L)")

        
horizons      = [1, 6, 12, 24]
marker_shapes = Dict(1=>:circle, 2=>:square, 3=>:diamond)

plots = Plots.Plot[]

for h in horizons
    p = plot(xlabel="#parameters", ylabel="Relative error", title="h = $(h)h")
    for L in L_list 
        xs = [data[(L,W)].params for W in W_list_fine]
        ys = [data[(L,W)].err[h] for W in W_list_fine]
        scatter!(p, xs, ys; markershape=marker_shapes[L], label="L = $L", markersize=6)
    end
    push!(plots, p)
end

plot(plots...;  layout=(2,2), 
                xscale=:log10, 
                plot_title="Relative Forecast Error vs. Model Size for Different Prediction Horizons", 
                size=(1000,800), margin=5mm, plot_margin=5mm)