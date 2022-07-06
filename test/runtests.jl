# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).

import Test
Test.@testset "Package EmpiricalDistributions" begin

# include("test_aqua.jl")
include("test_utils.jl")
include("test_uv_binned_dist.jl")
include("test_mv_binned_dist.jl")
include("test_docs.jl")

end # testset
