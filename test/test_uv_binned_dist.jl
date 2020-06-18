# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).

using EmpiricalDistributions
using Test

using Random
using Distributions, StatsBase


@testset "uv_binned_dist" begin
    Random.seed!(123)
    μ, σ = 1.23, 0.74
    true_dist = Normal(μ, σ)
    h = Histogram(μ-10σ:σ/10:μ+10σ)
    append!(h, rand(true_dist, 10^7))


    @test @inferred(UvBinnedDist(h)) isa UvBinnedDist
    d = UvBinnedDist(h)

    @test @inferred(convert(Histogram, d)) === d.hist

    @test @inferred(length(d)) == 1
    @test @inferred(size(d)) == ()
    @test @inferred(eltype(d)) == Float64

    @test all(isapprox.(mean(true_dist), @inferred(mean(d)), atol = 0.01))
    @test all(isapprox.(mode(true_dist), @inferred(mode(d)), atol = 0.05))
    @test all(isapprox.(var(true_dist), @inferred(var(d)), atol = 0.01))

    @test @inferred(minimum(d)) == μ-10σ
    @test @inferred(maximum(d)) == μ+10σ

    @test @inferred(rand(d)) isa Real
    @test @inferred(rand(d, 10)) isa Vector{<:Real}
    @test size(rand(d, 10)) == (10,)

    r = rand(d, 10^6)

    fit_dist = fit(Normal, r)
    @test isapprox(μ, fit_dist.μ, atol = 0.01)
    @test isapprox(σ, fit_dist.σ, atol = 0.01)

    @test @inferred(pdf(d, r[1])) isa Float64
    @test isapprox(pdf.(Ref(true_dist), r), pdf.(Ref(d), r), rtol = 0.1)
    @test @inferred(pdf(d, 1000)) == 0
    @test @inferred(pdf(d, -1000)) == 0
 
    @test @inferred(logpdf(d, r[1])) == log(pdf(d, r[1]))
    @test @inferred(logpdf(d, 1000)) == -Inf
    @test @inferred(logpdf(d, -1000)) == -Inf
end
