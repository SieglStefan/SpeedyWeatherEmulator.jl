# --- Bootstrap: docs-Umgebung + Paket einbinden ---
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()
# --------------------------------------------------

using SpeedyWeatherEmulator
using Documenter

# sorgt dafür, dass doctests im Kontext deines Pakets laufen
DocMeta.setdocmeta!(
    SpeedyWeatherEmulator,
    :DocTestSetup,
    :(using SpeedyWeatherEmulator);
    recursive = true,
)

# --- Dokumentation bauen ---
makedocs(
    modules  = [SpeedyWeatherEmulator],
    checkdocs = :exports,   # prüft nur Exports (empfohlen für API-Doku)
    authors  = "Siegl Stefan <stefan.siegl02@gmail.com>",
    sitename = "SpeedyWeatherEmulator.jl",
    format   = Documenter.HTML(
        canonical = "https://SieglStefan.github.io/SpeedyWeatherEmulator.jl",
        edit_link = "main",   # Branch-Name anpassen falls nötig
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
