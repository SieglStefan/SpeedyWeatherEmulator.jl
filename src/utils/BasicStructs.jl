module BasicStructs

export SimPara, NormStats, SimulationData

struct SimPara
    trunc::Int64
    n_steps::Int64
    n_ic::Int64
end

function SimPara(; trunc::Int64, n_steps::Int64, n_ic::Int64)
    return SimPara(trunc, n_steps, n_ic)
end

struct NormStats
    µ::Vector{Float32}
    σ::Vector{Float32}
end

struct SimulationData
    sim_para::SimPara
    data::Array{Float32, 3}
end


end