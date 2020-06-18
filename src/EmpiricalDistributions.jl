# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).

__precompile__(true)

module EmpiricalDistributions

using LinearAlgebra
using Random
using Statistics

using ArraysOfArrays
using Distributions
using StatsBase


include("hist_funcs.jl")
include("uv_binned_dist.jl")
include("mv_binned_dist.jl")

end # module
