# setup.jl

import Pkg

println("Creating a new environment and activating it...")
Pkg.activate(".")
Pkg.instantiate()

println("Adding required packages...")

# Core packages
packages = [
    "AutoEncoderToolkit",
    "Clustering",
    "Combinatorics",
    "CUDA",
    "CSV",
    "DataFrames",
    "DelimitedFiles",
    "Distributions",
    "Distances",
    "FileIO",
    "Flux",
    "GaussianMixtures",
    "Glob",
    "LinearAlgebra",
    "MAT",
    "Measures",
    "MultivariateStats",
    "NIfTI",
    "Plots",
    "Printf",
    "Random",
    "RegressionDynamicCausalModeling",
    "Statistics",
    "StatsBase",
    "TSne"
]

for pkg in packages
    try
        println("Installing $pkg...")
        Pkg.add(pkg)
    catch e
        @warn "Failed to install $pkg" exception=(e, catch_backtrace())
    end
end

println("Precompiling packages...")
Pkg.precompile()

println("Setup complete. Your environment is ready.")
