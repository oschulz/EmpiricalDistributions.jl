# EmpiricalDistributions.jl

A Julia package for empirical probability distributions that implement the
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl) API.

This package currently provides uni- and multivariate binned distributions
that can be created from
[StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl) histograms.

[`UvBinnedDist`](@ref), usually derived from a 1-dimensional histogram,
represents a binned univariate continuous distribution:

```julia
using Distributions, StatsBase

X_uv = rand(Normal(2.0, 0.5), 10^5)
uvhist = fit(Histogram, X_uv)

using EmpiricalDistributions

uvdist = UvBinnedDist(uvhist)
uvdist isa Distribution{Univariate,Continuous}
```

The resulting distribution can be queried, used to generate random numbers,
etc.:

```julia
mean(uvdist), var(uvdist)
maximum(uvdist), minimum(uvdist)
rand(uvdist, 5)
```

A binned distribution can be converted back to a histogram:

```julia
using LinearAlgebra
Histogram(uvdist) == normalize(uvhist)
```


[`MvBinnedDist`](@ref), usually derived from a multidimensional histogram,
represents a binned multivariate continuous distribution:

```julia
X_mv = rand(MvNormal([3.5, 0.5], [2.0 0.5; 0.5 1.0]), 10^5)
mvhist = fit(Histogram, (X_mv[1, :], X_mv[2, :]))

mvdist = MvBinnedDist(mvhist)
mvdist isa Distribution{Multivariate,Continuous}

mean(mvdist), cov(mvdist)
rand(mvdist, 5)
```
