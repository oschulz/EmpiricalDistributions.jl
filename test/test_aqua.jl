# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import EmpiricalDistributions

Test.@testset "Aqua tests" begin
    Aqua.test_all(EmpiricalDistributions)
end # testset
