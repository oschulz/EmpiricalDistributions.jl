# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).

using EmpiricalDistributions
using Test

using LinearAlgebra


@testset "utils" begin
    @testset "_bin_widths" begin
        @test @inferred(EmpiricalDistributions._bin_widths([2, 3, 5, 6, 10])) == [1, 2, 1, 4]
        @test @inferred(EmpiricalDistributions._bin_widths(2:2:10)) == fill(2, 4)
        @test @inferred(EmpiricalDistributions._bin_widths(2:1.5:10)) == fill(1.5, 5)
    end
end
