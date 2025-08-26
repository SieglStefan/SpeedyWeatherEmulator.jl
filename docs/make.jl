using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()


using SpeedyWeatherEmulator
using Documenter


DocMeta.setdocmeta!(
    SpeedyWeatherEmulator,
    :DocTestSetup,
    :(using SpeedyWeatherEmulator);
    recursive = true,
)


makedocs(
    modules  = [SpeedyWeatherEmulator],
    checkdocs = :exports, 
    authors  = "Siegl Stefan <stefan.siegl02@gmail.com>",
    sitename = "SpeedyWeatherEmulator.jl",
    format   = Documenter.HTML(
        canonical = "https://SieglStefan.github.io/SpeedyWeatherEmulator.jl",
        edit_link = "main",  
        assets    = String[],
    ),
    pages = [
        "Home"               => "index.md",
        "Quick installation" => "man/quickstart.md",
        "Theory"             => "man/theory.md",
        "Functions & Types"  => "man/api.md",
    ],
)

# --- Deployment nach GitHub Pages ---
deploydocs(
    repo      = "github.com/SieglStefan/SpeedyWeatherEmulator.jl",
    devbranch = "main",   # dein default branch
)
