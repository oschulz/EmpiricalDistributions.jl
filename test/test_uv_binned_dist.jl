# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).

using EmpiricalDistributions
using Test

using Random, LinearAlgebra
using Distributions, StatsBase
using Adapt, ForwardDiff


@testset "uv_binned_dist" begin
    μ, σ = 1.23, 0.74
    true_dist = Normal(μ, σ)
    h = Histogram(μ-10σ:σ/10:μ+10σ)
    append!(h, rand(true_dist, 10^7))

    @test @inferred(UvBinnedDist(h)) isa UvBinnedDist
    d = UvBinnedDist(h)
    @test @inferred(UvBinnedDist{Float32}(h)) isa UvBinnedDist{Float32}
    d_32 = UvBinnedDist{Float32}(h)
    @test @inferred(UvBinnedDist{Float64}(h)) isa UvBinnedDist{Float64}
    d_64 = UvBinnedDist{Float64}(h)

    @inferred(adapt(EmpiricalDistributions._AdaptToArray(), d))._edge isa Array

    @test @inferred(convert(Histogram, d)) == normalize(h)

    @test @inferred(length(d)) == 1
    @test @inferred(size(d)) == ()
    @test @inferred(eltype(d)) == Float64

    @test all(isapprox.(mean(true_dist), @inferred(mean(d)), atol = 0.01))
    @test all(isapprox.(mode(true_dist), @inferred(mode(d)), atol = 0.05))
    @test all(isapprox.(var(true_dist), @inferred(var(d)), atol = 0.01))

    @test @inferred(minimum(d)) == μ-10σ
    @test @inferred(maximum(d)) == μ+10σ

    @test @inferred(rand(d)) isa Real
    @test @inferred(rand(d_32)) isa Float32
    @test @inferred(rand(d_64)) isa Float64
    @test @inferred(rand(d, 10)) isa Vector{<:Real}
    @test @inferred(rand(d_32, 10)) isa Vector{Float32}
    @test @inferred(rand(d_64, 10)) isa Vector{Float64}
    @test size(rand(d, 10)) == (10,)

    r = rand(d, 10^6)

    fit_dist = fit(Normal, r)
    @test isapprox(μ, fit_dist.μ, atol = 0.01)
    @test isapprox(σ, fit_dist.σ, atol = 0.01)

    @test @inferred(pdf(d, r[1])) isa Real
    @test isapprox(pdf.(Ref(d), r), pdf.(Ref(true_dist), r), rtol = 0.1)
    @test @inferred(pdf(d, 1000)) == 0
    @test @inferred(pdf(d, -1000)) == 0
 
    @test @inferred(logpdf(d, r[1])) == log(pdf(d, r[1]))
    @test @inferred(logpdf(d, 1000)) == -Inf
    @test @inferred(logpdf(d, -1000)) == -Inf

    @test @inferred(cdf(d, r[1])) isa Real
    @test isapprox(cdf.(Ref(d), r), cdf.(Ref(true_dist), r), rtol = 0.1)
    @test @inferred(cdf(d, -1000)) == 0
    @test @inferred(cdf(d, 1000)) == 1

    diff_cdf(d::Distribution, x::Real) = ForwardDiff.derivative(a -> cdf(d, a), x)
    @test isapprox(diff_cdf.(Ref(d), r), pdf.(Ref(d), r), rtol = 0.1)

    @test @inferred(logcdf(d, r[1])) isa Real
    @test isapprox(logcdf.(Ref(d), r), logcdf.(Ref(true_dist), r), rtol = 0.1)
    @test @inferred(logcdf(d, -1000)) == -Inf
    @test @inferred(logcdf(d, 1000)) == 0

    @test @inferred(quantile(d, 0.5)) isa Real
    @test @inferred(quantile(d_32, 0.5f0)) isa Float32
    @test @inferred(quantile(d_64, 0.5)) isa Float64
    @test isapprox(quantile.(Ref(d), 0.05:0.01:0.95), quantile.(Ref(true_dist), 0.05:0.01:0.95), rtol = 0.1)
end
