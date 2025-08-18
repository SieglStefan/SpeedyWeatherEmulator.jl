module CompareEmulator

using SpeedyWeather, Statistics

using ..BasicStructs
using ..ModelStructs
using ..DataFormatting
using ..SimDataHandling


export compare_emulator


"""
    compare_emulator(   tm::TrainedModel; 
                        sim_data_comp::Sim_Data,
                        all_coeff::Bool=false)

Calculates and prints the relative error of the emulator `tm` relative to the forecast of SpeedyWeahter.jl.

# Arguments
- `tm::TrainedModel`:           Emulator which is compared to SpeedyWeather.jl.
- `sim_data_comp::Sim_Data`:    Simulation Data on which they are compared.
- `all_coeff::Bool = false`:    Switch for printing the rel. error of all coefficents.

# Returns
- `nothing`
"""
function compare_emulator(tm::TrainedModel; 
                            sim_data_comp::SimData,
                            all_coeff::Bool=false)
                            

    # Create formatted data with split_train=1.0, because there is no need of a validation set
    fd = FormattedData(sim_data_comp, split_train=1.0)

    # Create comparison vorticities
    vor_SW = fd.data_pairs.y_train              # Comparison vorticity from SpeedyWether.jl
    vor_EM = tm(fd.data_pairs.x_train)          # Testing vorticity from the Emulator
    
    # Calculate mean relative error
    rel_err = abs.(vor_EM .- vor_SW) ./ (abs.(vor_SW) .+ eps(Float32)) .* 100
    mean_rel_err = vec(mean(rel_err, dims=2))

    # Calculate the mean and max mean relative error of all spectral coefficents
    mean_mean_rel = mean(mean_rel_err)
    max_mean_rel = maximum(mean_rel_err)

    # Print results
    println("--------------------------------------")
    println("Mean relative error: ", round(mean_mean_rel; digits=3), " %")
    println("Max relative error:  ", round(max_mean_rel; digits=3), " %")
    println("--------------------------------------")

    if all_coeff
        for i in 1:length(mean_rel_err)
            # Some coefficients are always zero (in sim_data)
            if all(abs.(vor_SW[i, :]) .< eps(Float32))          
                println("coeff $i: rel. error = ",
                    round(mean_rel_err[i]; digits=3), " %, ", "\t (SW coeff. is always 0!!!)")
            else
                println("coeff $i: rel. error = ",
                    round(mean_rel_err[i]; digits=3), " %, ",)
            end
        end
        println("--------------------------------------")
    end

    return nothing
end


end

