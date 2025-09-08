using Test
using SpeedyWeatherEmulator
using Random


# Setting seed for reproducibility
Random.seed!(1234)

@testset "test_SpeedyWeatherEmulator" begin
    include("test_core.jl")
    include("test_io.jl")
    include("test_data.jl")
    include("test_emulator.jl")
    include("test_evaluation.jl")
    include("test_basic_workflow.jl")
end


