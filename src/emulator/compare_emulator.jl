using SpeedyWeather
using Statistics


"""
    compare_emulator(em::Emulator; 
                     x_test::Matrix{Float32},
                     y_test::Matrix{Float32},
                     n_it::Int64=1,
                     output::Bool=false,
                     all_coeff::Bool=false)

Compare emulator predictions against SpeedyWeather.jl reference data.

# Description
- Applies the emulator `em` to test inputs `x_test` and compares results to `y_test` for `n_it` timesteps.
- Computes mean relative error per spectral coefficient: 
    rel_err_i = |ŷ_i - y_i| / (|y_i| + ε) * 100.
- Prints mean relative error (all coefficients averaged) and maximum mean relative error.
- Optionally prints coefficient-wise relative errors.

# Arguments
- `em::Emulator`: Trained emulator to evaluate.
- `x_test::Matrix{Float32}`: Test inputs (vorticity coefficients at t) of form (2 * n_coeff, N).
- `y_test::Matrix{Float32}`: Reference outputs from SpeedyWeather.jl (at t+n_it*Δt) of form (2 * n_coeff, N).
- `n_it::Int64`: Number of timesteps compared.
- `output::Bool=false`: If true, print errors to STDOUT.
- `all_coeff::Bool=false`: If true, print relative error for each coefficient.

# Returns
- `mean_mean_rel::Float32`: The mean (all spectral coeff.) mean (all possible datapairs) relative error for `n_it` timesteps.

# Notes
- The larger `n_it`, the fewer data pairs are available for comparison. For example:
    `n_data=4` and `n_it=2` leads to data pairs 1-2-3 and 2-3-4.
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
                            n_it::Int64=1,
                            output::Bool=false,
                            all_coeff::Bool=false,
                            id_em::Bool=false)
    

    vor_em = x_test                        
    vor_sw = y_test

    # Create comparison vorticities
    n_del = em.sim_para.n_data - 1
    n_data_pairs = size(x_test,2)

    cols_skip_x = vcat([i+n_del-n_it+1:i+n_del-1 for i in 1:n_del:n_data_pairs]...)
    cols_skip_y = vcat([i:i+n_it-2 for i in 1:n_del:n_data_pairs]...)

    cols_delete_x = cols_skip_x[cols_skip_x .<= n_data_pairs]
    cols_delete_y = cols_skip_y[cols_skip_y .<= n_data_pairs]

    vor_em = vor_em[:, setdiff(1:end, cols_delete_x)]
    vor_sw = vor_sw[:, setdiff(1:end, cols_delete_y)]



    if id_em == false
        for _ in 1:n_it
            vor_em = em(vor_em)
        end
    end

    
    # Calculate mean relative error
    rel_err = abs.(vor_em .- vor_sw) ./ (abs.(vor_sw) .+ eps(Float32)) .* 100
    mean_rel_err = vec(mean(rel_err, dims=2))

    # Calculate the mean and max mean relative error of all spectral coefficents
    mean_mean_rel = mean(mean_rel_err)
    max_mean_rel = maximum(mean_rel_err)

    # Print results

    if output
        println("--------------------------------------")
        println("Mean relative error: ", round(mean_mean_rel; digits=3), " %")
        println("Max relative error:  ", round(max_mean_rel; digits=3), " %")
        println("--------------------------------------")

        # Prints mean relative error for every coefficient
        if all_coeff
            for i in 1:axis(mean_rel_err)
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
    end

    return mean_mean_rel
end


