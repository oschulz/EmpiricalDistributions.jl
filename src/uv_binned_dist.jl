# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).


"""
    UvBinnedDist <: Distribution{Univariate,Continuous}

Wraps a 1-dimensional histograms and presents it as a binned univariate
distribution.

Constructor:

    UvBinnedDist(h::Histogram{<:Real,1})
"""
struct UvBinnedDist{
    T <: Real,
    H <: Histogram{<:Real,1},
    VT <: AbstractVector{T}
} <: Distribution{Univariate,Continuous}
    hist::H
    _inv_weights::VT
    _volumes::VT
    _probabilty_edges::VT
    _probabilty_volumes::VT
    _probabilty_inv_volumes::VT
    _acc_prob::VT
    _mean::T
    _mode::T
    _var::T
end


export UvBinnedDist


function UvBinnedDist(h::Histogram{<:Real, 1}, T::DataType = Float64)
    nh = normalize(h)
    probabilty_widths::Vector{T} = h.weights * inv(sum(h.weights))
    probabilty_edges::Vector{T} = Vector{Float64}(undef, length(probabilty_widths) + 1)
    probabilty_edges[firstindex(probabilty_edges)] = 0
    @inbounds for (i, w) in enumerate(probabilty_widths)
        probabilty_edges[i+1] = probabilty_edges[i] + probabilty_widths[i]
    end
    probabilty_edges[end] = 1
    volumes = diff(h.edges[1])

    acc_prob::Vector{T} = zeros(T, length(nh.weights))
    for i in 2:length(acc_prob)
        acc_prob[i] += acc_prob[i-1] + nh.weights[i-1] * volumes[i-1]
    end

    mean_est = _mean(h)
    mode_est = _mode(nh)
    var_est = _var(h, mean = mean_est)

    return UvBinnedDist(
        nh,
        inv.(nh.weights),
        volumes,
        probabilty_edges,
        probabilty_widths,
        inv.(probabilty_widths),
        acc_prob,
        first(mean_est),
        first(mode_est),
        first(var_est)
    )
end


Base.convert(::Type{Histogram}, d::UvBinnedDist) = d.hist


Base.length(d::UvBinnedDist{T}) where {T} = 1
Base.size(d::UvBinnedDist{T}) where T = ()
Base.eltype(d::UvBinnedDist{T}) where {T} = T


Statistics.mean(d::UvBinnedDist{T, N}) where {T, N} = d._mean
StatsBase.mode(d::UvBinnedDist{T, N}) where {T, N} = d._mode
Statistics.var(d::UvBinnedDist{T, N}) where {T, N} = d._var
Statistics.cov(d::UvBinnedDist{T, N}) where {T, N} = d._cov

Distributions.minimum(d::UvBinnedDist) = first(d.hist.edges[1])
Distributions.maximum(d::UvBinnedDist) = last(d.hist.edges[1])

Distributions.pdf(d::UvBinnedDist, x::Real) = _pdf(d.hist, (x,))

Distributions.logpdf(d::UvBinnedDist, x::Real) = log(pdf(d, x))


# ToDo: Efficient implementation, should cache CDF
function Distributions.cdf(d::UvBinnedDist, x::Real)
    i::Int = StatsBase.binindex(d.hist, x)
    p::T = @inbounds sum(d.hist.weights[1:i-1] .* d._volumes[1:i-1])
    p += (x - d.hist.edges[1][i]) * d.hist.weights[i]
    return p
end


function Distributions.quantile(d::UvBinnedDist{T}, x::Real)::T where {T <: AbstractFloat}
    r::UnitRange{Int} = searchsorted(d._acc_prob, x)
    idx::Int = min(r.start, r.stop)
    p::T = d._acc_prob[ idx ]
    q::T = d.hist.edges[1][idx]
    missing_p::T = x - p
    inv_weight::T = d._inv_weights[idx]
    if !isinf(inv_weight)
        q += missing_p * inv_weight
    end
    return min(q, maximum(d))
end


function Random.rand(rng::AbstractRNG, d::UvBinnedDist{T})::T where {T <: AbstractFloat}
    r::T = rand()
    next_inds::UnitRange{Int} = searchsorted(d._probabilty_edges, r)
    next_ind_l::Int = next_inds.start
    next_ind_r::Int = next_inds.stop
    if next_ind_l > next_ind_r
        next_ind_l = next_inds.stop
        next_ind_r = next_inds.start
    end
    ret::T = d.hist.edges[1][next_ind_l]
    if next_ind_l < next_ind_r
        ret += d._volumes[next_ind_l] * (d._probabilty_edges[next_ind_r] - r) * d._probabilty_inv_volumes[next_ind_l]
    end
    return ret
end
