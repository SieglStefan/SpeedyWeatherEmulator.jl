module CompareEmulator

using JLD2, SpeedyWeather, Statistics, Printf

using ..BasicStructs
using ..ModelStructs
using ..DataFormatting
using ..SimDataHandling

export compare_emulator



function compare_emulator(tm::TrainedModel; all_coeff::Bool=false, sim_para::SimPara = tm.sim_para, initial_cond::Union{Nothing,Function}=nothing)


    sim_para = SimPara(trunc = sim_para.trunc, n_steps = sim_para.n_steps, n_ic=sim_para.n_ic)

    sim_data = create_sim_data(sim_para, initial_cond=initial_cond)
    fd = FormattedData(sim_data, split=1.0)

    vor_NN = tm(fd.data_pairs.x_train)
    vor_SW = fd.data_pairs.y_train

    rel_err = abs.(vor_NN .- vor_SW) ./ (abs.(vor_SW) .+ eps(Float32)) .* 100
    mean_rel_err = vec(mean(rel_err, dims=2))

    # Zusammenfassende Werte
    mean_mean_rel = mean(mean_rel_err)
    max_mean_rel = maximum(mean_rel_err)

    println("--------------------------------------")
    println("Mean relative error: ", round(mean_mean_rel; digits=3), " %")
    println("Max relative error:  ", round(max_mean_rel; digits=3), " %")
    println("--------------------------------------")

    if all_coeff
        for i in 1:length(mean_rel_err)
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
end


end

