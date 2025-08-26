using SpeedyWeather
using Statistics


"""
    compare_emulator(em::Emulator; 
                     x_test::Matrix{Float32},
                     y_test::Matrix{Float32},
                     all_coeff::Bool=false)

Compare emulator predictions against SpeedyWeather.jl reference data.

# Description
- Applies the emulator `em` to test inputs `x_test` and compares results to `y_test`.
- Computes mean relative error per spectral coefficient: 
    rel_err_i = |ŷ_i - y_i| / (|y_i| + ε) * 100.
- Prints mean relative error (all coefficients averaged) and maximum mean relative error.
- Optionally prints coefficient-wise relative errors.

# Arguments
- `em::Emulator`: Trained emulator to evaluate.
- `x_test::Matrix{Float32}`: Test inputs (vorticity coefficients at t) of form (2 * n_coeff, N).
- `y_test::Matrix{Float32}`: Reference outputs from SpeedyWeather.jl (at t+Δt) of form (2 * n_coeff, N).
- `all_coeff::Bool=false`: If true, print relative error for each coefficient.

# Returns
- `nothing`: Results are printed to STDOUT.

# Notes
- Some coefficients in SpeedyWeather.jl are structurally zero → flagged in output.
- Errors are reported in percent [%].

# Examples
```julia
emu, losses = train_emulator(nn, fd)
compare_emulator(emu; 
    x_test=fd.data_pairs.x_test,
    y_test=fd.data_pairs.y_test,
    all_coeff=true)
```
"""
function compare_emulator(em::Emulator; 
                            x_test::Matrix{Float32},
                            y_test::Matrix{Float32},
                            all_coeff::Bool=false)
                            

    # Create comparison vorticities
    vor_sw = y_test              # Comparison vorticity from SpeedyWeather.jl
    vor_em = em(x_test)          # Testing vorticity from the Emulator
    
    # Calculate mean relative error
    rel_err = abs.(vor_em .- vor_sw) ./ (abs.(vor_sw) .+ eps(Float32)) .* 100
    mean_rel_err = vec(mean(rel_err, dims=2))

    # Calculate the mean and max mean relative error of all spectral coefficents
    mean_mean_rel = mean(mean_rel_err)
    max_mean_rel = maximum(mean_rel_err)

    # Print results
    println("--------------------------------------")
    println("Mean relative error: ", round(mean_mean_rel; digits=3), " %")
    println("Max relative error:  ", round(max_mean_rel; digits=3), " %")
    println("--------------------------------------")

    # Prints mean relative error for every coefficient
    if all_coeff
        for i in 1:length(mean_rel_err)
            # Some coefficients are always zero (in sim_data)
            if all(abs.(vor_sw[i, :]) .< eps(Float32))          
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

