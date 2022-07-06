# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).

using Test
using EmpiricalDistributions
import Documenter

Documenter.DocMeta.setdocmeta!(
    EmpiricalDistributions,
    :DocTestSetup,
    :(using EmpiricalDistributions);
    recursive=true,
)
Documenter.doctest(EmpiricalDistributions)
