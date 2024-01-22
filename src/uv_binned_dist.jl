# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).


"""
    UvBinnedDist{T<:Real} <: StatsBase.Distribution{Univariate,Continuous}

A binned univariate distribution, usually derived from a histogram.

Constructors:

```julia
    UvBinnedDist(h::StatsBase.Histogram{<:Real,1})
    UvBinnedDist{T<:Real}(h::StatsBase.Histogram{<:Real,1})
```

You can convert a `UvBinnedDist` back to a histogram via 

```julia
convert(StatsBase.Histogram, dist::UvBinnedDist)
StatsBase.Histogram(dist::UvBinnedDist)
```
"""
struct UvBinnedDist{
    T <: Real,
    U <: Real,
    VT <: AbstractVector{T},
    VU <: AbstractVector{U}
} <: Distribution{Univariate,Continuous}
    _edge::VT
    _edge_cdf::VU
    _bin_pdf::VU
    _bin_probmass::VU
    _closed_left::Bool
    _mean::T
    _mode::T
    _var::T
end


export UvBinnedDist


function UvBinnedDist{T}(h::Histogram{<:Real,1}) where {T<:Real}
    nh = normalize(h)

    edge = T.(first(nh.edges))
    closed_left = nh.closed == :left

    bin_pdf = nh.weights 
    bin_probmass = bin_pdf .* _bin_widths(edge)
    bin_probmass .*= inv(sum(bin_probmass))

    edge_cdf = pushfirst!(cumsum(bin_probmass), 0)
    @assert last(edge_cdf) ≈ 1
    edge_cdf[end] = 1

    mean_est_tpl = _mean(nh)
    mean_est = T(first(mean_est_tpl))
    mode_est = T(first(_mode(nh)))
    var_est = T(first(_var(nh, mean_est_tpl)))

    return UvBinnedDist(
        edge, edge_cdf, bin_pdf, bin_probmass, closed_left,
        mean_est, mode_est, var_est
    )
end

UvBinnedDist(h::Histogram{<:Real,1}) = UvBinnedDist{float(eltype(first(h.edges)))}(h)


function Adapt.adapt_structure(to, d::UvBinnedDist)
    UvBinnedDist(
        adapt(to, d._edge), adapt(to, d._edge_cdf),
        adapt(to, d._bin_pdf), adapt(to, d._bin_probmass),
        d._closed_left, d._mean, d._mode, d._var
    )
end


Histogram(d::UvBinnedDist) = Histogram((Array(d._edge),), Array(d._bin_pdf), (d._closed_left ? :left : :right), true)
Base.convert(::Type{Histogram}, d::UvBinnedDist) = Histogram(d)


Base.length(d::UvBinnedDist) = 1
Base.size(d::UvBinnedDist) = ()
Base.eltype(d::UvBinnedDist{T}) where {T} = T

Distributions.params(d::UvBinnedDist) = (d._edge, d._bin_pdf, d._closed_left)

Statistics.mean(d::UvBinnedDist) = d._mean
StatsBase.mode(d::UvBinnedDist) = d._mode
Statistics.var(d::UvBinnedDist) = d._var

Distributions.minimum(d::UvBinnedDist) = first(d._edge)
Distributions.maximum(d::UvBinnedDist) = last(d._edge)


function Distributions.pdf(d::UvBinnedDist{T,U}, x::Real) where {T,U}
    if insupport(d, x)
        i = _find_bin(d._edge, d._closed_left, x)
        i <= lastindex(d._bin_pdf) ? convert(U, d._bin_pdf[i]) : zero(U)
    else
        zero(U)
    end
end

Distributions.logpdf(d::UvBinnedDist, x::Real) = log(pdf(d, x))


function Distributions.cdf(d::UvBinnedDist{T}, x::Real) where T
    if x <= minimum(d)
        zero(T)
    elseif x >= maximum(d)
        one(T)
    else
        _linear_interpol(d._edge, d._edge_cdf, x)
    end
end


function Distributions.quantile(d::UvBinnedDist, x::Real)
    0 <= x <= 1 || throw(DomainError(x, "quantile requires value between 0 and 1"))
    _linear_interpol(d._edge_cdf, d._edge, x)
end


function Random.rand(rng::AbstractRNG, d::UvBinnedDist{T}) where T
    u = rand(T)
    @assert axes(d._edge) == axes(d._edge_cdf)
    i_lo, i_hi = _find_idxs_lohi(d._edge_cdf, u)
    x_lo, x_hi = d._edge[i_lo], d._edge[i_hi]
    _rand_uniform(rng, T, x_lo, x_hi)
end
