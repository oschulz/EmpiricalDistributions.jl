# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).

using EmpiricalDistributions
using Test

using Random, LinearAlgebra
using Distributions, StatsBase, ArraysOfArrays


@testset "mv_binned_dist" begin
    Random.seed!(123)
    μ = [1.23, -0.67]
    Σ = [0.45 0.32; 0.32 0.76]' * [0.45 0.32; 0.32 0.76]
    true_dist = MvNormal(μ, Σ)
    h = Histogram((μ[1]-10Σ[1]:Σ[1]/10:μ[1]+10Σ[1], μ[2]-10Σ[4]:Σ[4]/10:μ[2]+10Σ[4]))
    n = 10^6
    r = rand(true_dist, n)
    append!(h, (r[1, :], r[2, :]))

    @test @inferred(MvBinnedDist(h)) isa MvBinnedDist
    d = MvBinnedDist(h)

    @test @inferred(convert(Histogram, d)) === d.hist

    @test @inferred(length(d)) == 2
    @test @inferred(size(d)) == (2,)
    @test @inferred(eltype(d)) == Float64

    @test all(isapprox.(mean(true_dist), @inferred(mean(d)),   atol = 0.01))
    @test all(isapprox.(mode(true_dist), @inferred(mode(d)),   atol = 0.05))
    @test all(isapprox.(var(true_dist), @inferred(var(d)),   atol = 0.01))
    @test all(isapprox.(cov(true_dist), @inferred(cov(d)),   atol = 0.01))

    @test @inferred(rand!(d, r)) === r

    fit_dist = fit(MvNormal, r)
    @test all(isapprox.(μ, fit_dist.μ,     atol = 0.01))
    @test all(isapprox.(Σ, fit_dist.Σ.mat, atol = 0.01))

    @test @inferred(pdf(d, r[:,1])) isa Float64
    @test isapprox(pdf.(Ref(true_dist), nestedview(r)), pdf.(Ref(d), nestedview(r)), rtol = 0.1)
    @test @inferred(pdf(d, [1000, 1000])) == 0
    @test @inferred(pdf(d, [-1000, -1000])) == 0
 
    @test @inferred(logpdf(d, r[:,1])) == log(pdf(d, r[:,1]))
    @test @inferred(logpdf(d, [1000, 1000])) == -Inf
    @test @inferred(logpdf(d, [-1000, -1000])) == -Inf

    @test @inferred(rand(d)) isa Vector{Float64}
    @test size(rand(d)) == (length(d),)

    @test @inferred(rand(d, 10)) isa Matrix{Float64}
    @test size(rand(d, 10)) == (length(d), 10)
end
