# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).


"""
    MvBinnedDist <: Distribution{Multivariate,Continuous}

Wraps a multi-dimensional histograms and presents it as a binned multivariate
distribution.

Constructor:

    MvBinnedDist(h::Histogram{<:Real,N})
"""
struct MvBinnedDist{
    T <: Real,
    N,
    H <: Histogram{<:Real, N},
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T}
} <: Distributions.Distribution{Multivariate,Continuous}
    hist::H
    _edges::NTuple{N, <:AbstractVector{T}}
    _cart_inds::CartesianIndices{N, NTuple{N, Base.OneTo{Int}}}
    _probability_edges::VT
    _mean::VT
    _mode::VT
    _var::VT
    _cov::MT
end

export MvBinnedDist


function MvBinnedDist(h::StatsBase.Histogram{<:Real, N}, T::DataType = Float64) where {N}
    nh = normalize(h)

    probabilty_widths = nh.weights * inv(sum(nh.weights))
    probabilty_edges::Vector{T} = Vector{Float64}(undef, length(h.weights) + 1)
    probabilty_edges[1] = 0
    for (i, w) in enumerate(probabilty_widths)
        v = probabilty_edges[i] + probabilty_widths[i]
        probabilty_edges[i+1] = v > 1 ? 1 : v
    end

    mean_est = _mean(h)
    mode_est = _mode(nh)
    var_est = _var(h, mean = mean_est)
    cov_est = _cov(h, mean = mean_est)

    return MvBinnedDist(
        nh,
        collect.(nh.edges),
        CartesianIndices(nh.weights),
        probabilty_edges,
        mean_est,
        mode_est,
        var_est,
        cov_est
    )
end


Base.convert(::Type{Histogram}, d::MvBinnedDist) = d.hist


Base.length(d::MvBinnedDist{T, N}) where {T, N} = N
Base.size(d::MvBinnedDist{T, N}) where {T, N} = (N,)
Base.eltype(d::MvBinnedDist{T, N}) where {T, N} = T

Statistics.mean(d::MvBinnedDist{T, N}) where {T, N} = d._mean
StatsBase.mode(d::MvBinnedDist{T, N}) where {T, N} = d._mode
Statistics.var(d::MvBinnedDist{T, N}) where {T, N} = d._var
Statistics.cov(d::MvBinnedDist{T, N}) where {T, N} = d._cov


function Distributions._rand!(r::AbstractRNG, d::MvBinnedDist{T,N}, A::AbstractVector{<:Real}) where {T, N}
    rand!(r, A)
    next_inds::UnitRange{Int} = searchsorted(d._probability_edges::Vector{T}, A[1]::T)
    cell_lin_index::Int = min(next_inds.start, next_inds.stop)
    cell_car_index = d._cart_inds[cell_lin_index]
    for idim in Base.OneTo(N)
        i = cell_car_index[idim]
        sub_int = d._edges[idim][i:i+1]
        sub_int_width::T = sub_int[2] - sub_int[1]
        A[idim] = sub_int[1] + sub_int_width * A[idim]
    end
    return A
end

function Distributions._rand!(r::AbstractRNG, d::MvBinnedDist{T,N}, A::AbstractMatrix{<:Real}) where {T, N}
    Distributions._rand!.((r,), (d,), nestedview(A))
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


function Distributions.pdf(d::MvBinnedDist{T,N}, x::AbstractVector{<:Real}) where {T,N}
    length(eachindex(x)) == N || throw(ArgumentError("Length of variate doesn't match dimensionality of distribution"))
    x_tpl = _unsafe_unroll_tuple(x, Val(N))
    _pdf(d.hist, x_tpl)
end


function Distributions.logpdf(d::MvBinnedDist{T, N}, x::AbstractArray{<:Real, 1}) where {T, N}
    return log(pdf(d, x))
end

function Distributions._logpdf(d::MvBinnedDist{T,N}, x::AbstractArray{<:Real, 1}) where {T, N}
    return logpdf(d, x)
end
