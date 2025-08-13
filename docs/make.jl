using SpeedyWeatherEmulator
using Documenter

DocMeta.setdocmeta!(SpeedyWeatherEmulator, :DocTestSetup, :(using SpeedyWeatherEmulator); recursive=true)

makedocs(;
    modules=[SpeedyWeatherEmulator],
    authors="Siegl Stefan stefan.siegl02@gmail.com",
    sitename="SpeedyWeatherEmulator.jl",
    format=Documenter.HTML(;
        canonical="https://SieglStefan.github.io/SpeedyWeatherEmulator.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/SieglStefan/SpeedyWeatherEmulator.jl",
    devbranch="main",
)
