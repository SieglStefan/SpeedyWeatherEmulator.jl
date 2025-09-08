using SpeedyWeatherEmulator
using Plots, Measures


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

sim_data.data


# Different parameters for the fine optimization
L_list = [1]                                                                # number of hidden layers
W_list_fine = [384, 512, 640, 768, 896, 1024, 1280, 1536, 2048]        # number of neurons per hidden layer                                      # number of neurons per hidden layer (coarse)                        


d = 2*calc_n_coeff(trunc=TRUNC)                                
n_params(d, W, L) = (d*W + W) + (L-1)*(W*W + W) + (W*d + d)
plot_data = Dict{Tuple{Int,Int}, NamedTuple}()  
horizons = [1, 6, 12, 24]                  


for L in L_list, W in W_list_fine

    id = "_hyperpara_L$(L)_W$(W)"
    sim_para = SimPara(trunc=TRUNC, n_data=N_DATA, n_ic=N_IC, id_key=id)

    em = load_data(sim_para, type="emulator")

    err_vec = zeros(N_DATA)

    for steps in horizons
        err_vec[steps] = compare_emulator(  em,
                                            x_test=fd.data_pairs.x_valid,
                                            y_test=fd.data_pairs.y_valid,
                                            n_it=steps)

    end

    plot_data[(L,W)] = (err=err_vec, params=n_params(d, W, L))
end


# Small differences in the plotting scheme     
marker_shapes = Dict(1=>:circle, 2=>:square, 3=>:diamond)
plots = Plots.Plot[]

for h in horizons
    p = Plots.plot( xlabel="Number of parameters", 
                    ylabel="Relative error / %",
                    title="horizon = $(h)h", 
                    titlefontsize=18)                     
    for L in L_list 
        xs = [plot_data[(L,W)].params for W in W_list_fine]
        ys = [plot_data[(L,W)].err[h] for W in W_list_fine]
        
        Plots.scatter!(p, xs, ys;
                 markershape=marker_shapes[L],
                 label="",
                 markersize=8)                             
        
        # Create offset for W index
        yspan = maximum(ys) - minimum(ys)
        dy    = 0.04 * (yspan == 0 ? 1 : yspan)            

        for (xi, yi, W) in zip(xs, ys, W_list_fine)
            annotate!(p, xi, yi + dy, Plots.text("$(W)", 11, :black, :center, :bottom))
        end
    end
    push!(plots, p)
end


p = Plots.plot(plots...; layout=(2,2),
                    plot_title="Results of Fine Hyperparameter Optimization",
                    size=(1000,900), margin=8mm,
                    xscale=:log10, 
                    xticks = ([5e4, 1e5, 2e5], ["5×10⁴", "10⁵", "2×10⁵"]), 
                    plot_titlefontsize=25,           
                    guidefont=13,                
                    tickfont=12,                       
                    legendfontsize=12)                 


display(p)
Plots.savefig(p, joinpath(@__DIR__, "plots", "fine_opt.pdf"))