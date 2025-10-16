# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).


"""
    MvBinnedDist <: StatsBase.Distribution{Multivariate,Continuous}

A binned multivariate distribution, usually derived from a histogram.

Constructors:

```julia
    MvBinnedDist(h::StatsBase.Histogram{<:Real,N})
    MvBinnedDist{T<:Real}(h::StatsBase.Histogram{<:Real,N})
```

You can convert a `MvBinnedDist` back to a histogram via 

```julia
convert(StatsBase.Histogram, dist::MvBinnedDist)
StatsBase.Histogram(dist::MvBinnedDist)
```
"""
struct MvBinnedDist{
    T <: Real,
    N,
    U <: Real,
    ET <: NTuple{N,AbstractVector{<:Real}},
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    VU <: AbstractVector{U},
    AU <: AbstractArray{U,N}
} <: Distributions.Distribution{Multivariate,Continuous}
    _edges::ET
    _bin_pdf::AU
    _bin_linidx_cdf::VU
    _closed_left::Bool
    _mean::VT
    _mode::VT
    _var::VT
    _cov::MT
end

export MvBinnedDist


function MvBinnedDist{T}(h::Histogram{<:Real}) where {T<:Real}
    nh = normalize(h)

    edges = nh.edges
    bin_pdf = nh.weights 

    closed_left = nh.closed == :left

    Y = nh.weights
    X = _bin_centers.(nh.edges)
    W = _bin_widths.(nh.edges)

    bin_linidx_cdf = cumsum(broadcast(idx -> Y[idx] .* prod(map(getindex, W, idx.I)), vec(CartesianIndices(Y))))
    @assert last(bin_linidx_cdf) ≈ 1
    bin_linidx_cdf[end] = 1

    mean_est_tpl = _mean(nh)
    mean_est = [T.(mean_est_tpl)...]
    mode_est = [T.(_mode(nh))...]
    var_est = [T.(_var(nh, mean_est_tpl))...]
    cov_est = T.(_cov(nh, mean_est_tpl))

    return MvBinnedDist(
        edges, bin_pdf, bin_linidx_cdf, closed_left,
        mean_est, mode_est, var_est, cov_est
    )
end

MvBinnedDist(h::Histogram{<:Real}) = MvBinnedDist{float(promote_type(map(eltype, h.edges)...))}(h)


function Adapt.adapt_structure(to, d::MvBinnedDist)
    MvBinnedDist(
        map(e -> adapt(to, e), d._edges), adapt(to, d._bin_pdf), adapt(to, d._bin_linidx_cdf),
        adapt(to, d._closed_left), adapt(to, d._mean), adapt(to, d._mode), adapt(to, d._var), adapt(to, d._cov)
    )
end


StatsBase.Histogram(d::MvBinnedDist) = Histogram(map(Array, d._edges), Array(d._bin_pdf), (d._closed_left ? :left : :right), true)
Base.convert(::Type{Histogram}, d::MvBinnedDist) = Histogram(d)


Base.length(d::MvBinnedDist{T,N}) where {T,N} = N
Base.size(d::MvBinnedDist{T,N}) where {T,N} = (N,)
Base.eltype(d::MvBinnedDist{T,N}) where {T,N} = T

Distributions.params(d::MvBinnedDist) = (d._edges, d._bin_pdf, d._closed_left)

Statistics.mean(d::MvBinnedDist) = d._mean
StatsBase.mode(d::MvBinnedDist) = d._mode
Statistics.var(d::MvBinnedDist) = d._var
Statistics.cov(d::MvBinnedDist) = d._cov


function Distributions._rand!(rng::AbstractRNG, d::MvBinnedDist{T,N}, A::AbstractVector{<:Real}) where {T,N}
    @assert length(eachindex(A)) == N
    u = rand(rng)
    i = searchsortedfirst(d._bin_linidx_cdf, u)
    idx_lo = CartesianIndices(d._bin_pdf)[i]
    idx_hi = idx_lo + CartesianIndex(ntuple(i -> 1, Val(N))...)
    x_lo = map(getindex, d._edges, idx_lo.I)
    x_hi = map(getindex, d._edges, idx_hi.I)
    for i in 1:N
        A[i] = _rand_uniform(rng, T, x_lo[i], x_hi[i])
    end
    return A
end

function Distributions._rand!(rng::AbstractRNG, d::MvBinnedDist{T,N}, A::AbstractMatrix{<:Real}) where {T,N}
    Distributions._rand!.(Ref(rng), (d,), nestedview(A))
    return A
end


# Similar to unroll_tuple in StaticArrays.jl:
@generated function _unsafe_unroll_tuple(A::AbstractArray, ::Val{L}) where {L}
    exprs = [:(A[idx0 + $j]) for j = 0:(L-1)]
    quote
        idx0 = firstindex(A)
        Base.@_inline_meta
        @inbounds return $(Expr(:tuple, exprs...))
    end
end


function Distributions.pdf(d::MvBinnedDist{T,N,U}, x::AbstractVector{<:Real}) where {T,N,U}
    length(eachindex(x)) == N || throw(ArgumentError("Length of variate doesn't match dimensionality of distribution"))
    xs = _unsafe_unroll_tuple(x, Val(N))

    idxs = _find_bin(d._edges, d._closed_left, xs)
    if checkbounds(Bool, d._bin_pdf, idxs...)
        @inbounds r = d._bin_pdf[idxs...]
        convert(U, r)
    else
        zero(U)
    end
end


function Distributions.logpdf(d::MvBinnedDist{T,N}, x::AbstractArray{<:Real, 1}) where {T,N}
    return log(pdf(d, x))
end

function Distributions._logpdf(d::MvBinnedDist{T,N}, x::AbstractArray{<:Real, 1}) where {T,N}
    return logpdf(d, x)
end
