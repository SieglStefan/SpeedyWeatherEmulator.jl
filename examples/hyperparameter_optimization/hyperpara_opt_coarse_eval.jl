using SpeedyWeatherEmulator
using Plots
using ProgressMeter
using Measures

n_params(d, W, L) = (d*W + W) + (L-1)*(W*W + W) + (W*d + d)

const TRUNC = 5
const N_DATA = 48
const N_IC = 1000

sim_para_loading = SimPara(trunc=TRUNC, n_data=N_DATA, n_ic=N_IC)
sim_data = load_data(sim_para_loading, type="sim_data")
fd = FormattedData(sim_data)

x_valid = fd.data_pairs.x_valid
y_valid = fd.data_pairs.y_valid

L_list = [1,2,3]
W_list = [64, 128, 256, 512]
d = 2*calc_n_coeff(trunc=TRUNC)

data = Dict{Tuple{Int,Int}, NamedTuple}()

@showprogress for L in L_list, W in W_list

    id = "_hyperpara_L$(L)_W$(W)"
    sim_para = SimPara(trunc=TRUNC, n_data=N_DATA, n_ic=N_IC, id_key=id)

    em = load_data(sim_para, type="emulator")
    losses = load_data(sim_para, type="losses")

    err_vec = zeros(N_DATA)

    for steps in 1:N_DATA
        err_vec[steps] = compare_emulator(  em,
                                            x_test=x_valid,
                                            y_test=y_valid,
                                            n_it=steps)
    end

    data[(L,W)] = (
        err = err_vec,
        training_time = losses.training_time,
        params = n_params(d, W, L)
    )

end


for L in L_list, W in W_list

    err = data[(L,W)].err
    time = data[(L,W)].training_time
    params = data[(L,W)].params

    println("L$(L)_W$(W): ", err[3], ", ", time, ", ", params)

end

params = [data[(L,W)].params for L in L_list for W in W_list]
times  = [data[(L,W)].training_time for L in L_list for W in W_list]

colors = [L for L in L_list for W in W_list]   # färbt nach L
scatter(params, times;
        group=colors,
        xlabel="Number of parameters",
        ylabel="Training time [s]",
        title="Training time vs. #params (colored by L)")


horizons      = [1, 6, 12, 24]
marker_shapes = Dict(1=>:circle, 2=>:square, 3=>:diamond)

plots = Plots.Plot[]  # typisiert

for h in horizons
    p = plot(xlabel="#parameters", ylabel="Relative error", title="h = $(h)h")
    for L in L_list 
        xs = [data[(L,W)].params for W in W_list]
        ys = [data[(L,W)].err[h] for W in W_list]
        scatter!(p, xs, ys; markershape=marker_shapes[L], label="L = $L", markersize=6)
    end
    push!(plots, p)  # <- NUR HIER, EINMAL pro h
end

plot(plots...; layout=(2,2), xscale=:log10, plot_title="Relative Forecast Error vs. Model Size for Different Prediction Horizons", size=(1000,800),margin=5mm, plot_margin=50mm)