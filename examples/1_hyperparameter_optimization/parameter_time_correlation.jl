using SpeedyWeatherEmulator
using Plots, Measures


### Prepare emulator evaluation

# Define simulation parameters
const TRUNC = 5
const N_DATA = 48
const N_IC = 1000


# Define the parameters for the coarse hyperparameter optimization
L_list_coarse = [1,2,3]
L_list_fine = [1]
W_list_coarse = [64, 128, 256, 512]     # number of neurons per hidden layer
W_list_fine = [256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048]        # number of neurons per hidden layer

pairs_coarse = [(L, W) for L in L_list_coarse for W in W_list_coarse]
pairs_fine   = [(L, W) for L in L_list_fine for W in W_list_fine]
pairs = vcat(pairs_coarse, pairs_fine)


# Calculate number of spectral coeff.
d = 2*calc_n_coeff(trunc=TRUNC)                                

# Calculate the overall number of parameters of the neural network
n_params(d, W, L) = (d*W + W) + (L-1)*(W*W + W) + (W*d + d)

# Define dict. for better accessing training time and number of network parameters
plot_data = Dict{Tuple{Int,Int}, NamedTuple}()                     


# Fill the data dict. for all coarse hyperparameter
for (L,W) in pairs

    # Define simulation parameters for specific hyperparameter
    id = "_hyperpara_L$(L)_W$(W)"
    sim_para = SimPara(trunc=TRUNC, n_data=N_DATA, n_ic=N_IC, id_key=id)

    # Load the specific emulator losses
    losses = load_data(Losses, sim_para)

    # Initialize the dict.
    plot_data[(L,W)] = (training_time=losses.training_time, params = n_params(d, W, L))
end




### Plot the correlation of training time and number of neural network parameters 

# Define the neural network parameters and training times as lists
params = [plot_data[(L,W)].params for (L,W) in pairs]
times  = [plot_data[(L,W)].training_time for (L,W) in pairs]

# Define different colors for different L values
colors = [L for (L,W) in pairs]

# Plot the correlation
p = Plots.scatter(params, times./60;
                    group=[ "L = $(L)" for L in colors ],
                    title="Training Time vs. Number of Parameters",
                    xlabel="Number of parameters",
                    ylabel="Training time / min",
                    xticks = (0:1e5:5e5, ["0","10⁵","2×10⁵","3×10⁵","4×10⁵","5×10⁵"]),
                    xformatter = x -> "$(Int(round(x/1e5)))×10⁵",
                    plot_titlefontsize=25,              # title size
                    guidefont=13,                       # axis title size
                    tickfont=12,                        # tick size
                    legendfontsize=12,                  # legend size
                    markersize=8)                       # marker size


# Display and save image                
display(p)
Plots.savefig(p, joinpath(@__DIR__, "plots", "parameter_vs_time.pdf"))