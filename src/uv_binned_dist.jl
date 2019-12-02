# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).


struct UvBinnedDist{T <: AbstractFloat} <: Distribution{Univariate,Continuous}
    h::Histogram{<:Real, 1}
    inv_weights::Vector{T}
    edges::Vector{T}
    volumes::Vector{T}

    probabilty_edges::Vector{T}
    probabilty_volumes::Vector{T}
    probabilty_inv_volumes::Vector{T}

    acc_prob::Vector{T}

    μ::T
    var::T
    cov::Matrix{T}
    σ::T
end

export UvBinnedDist


function UvBinnedDist(h::Histogram{<:Real, 1}, T::DataType = Float64)
    nh = normalize(h)
    probabilty_widths::Vector{T} = h.weights * inv(sum(h.weights))
    probabilty_edges::Vector{T} = Vector{Float64}(undef, length(probabilty_widths) + 1)
    probabilty_edges[1] = 0
    @inbounds for (i, w) in enumerate(probabilty_widths)
        probabilty_edges[i+1] = probabilty_edges[i] + probabilty_widths[i]
    end
    probabilty_edges[end] = 1
    volumes = diff(h.edges[1])
    mean = Statistics.mean(StatsBase.midpoints(nh.edges[1]), ProbabilityWeights(nh.weights))
    var = Statistics.var(StatsBase.midpoints(nh.edges[1]), ProbabilityWeights(nh.weights), mean = mean)

    acc_prob::Vector{T} = zeros(T, length(nh.weights))
    for i in 2:length(acc_prob)
        acc_prob[i] += acc_prob[i-1] + nh.weights[i-1] * volumes[i-1]
    end

    d::UvBinnedDist{T} = UvBinnedDist{T}(
        nh,
        inv.(nh.weights),
        nh.edges[1],
        volumes,
        probabilty_edges,
        probabilty_widths,
        inv.(probabilty_widths),
        acc_prob,
        mean,
        var,
        fill(var, 1, 1),
        sqrt(var)
    )
end


function Random.rand(rng::AbstractRNG, d::UvBinnedDist{T})::T where {T <: AbstractFloat}
    r::T = rand()
    next_inds::UnitRange{Int} = searchsorted(d.probabilty_edges, r)
    next_ind_l::Int = next_inds.start
    next_ind_r::Int = next_inds.stop
    if next_ind_l > next_ind_r
        next_ind_l = next_inds.stop
        next_ind_r = next_inds.start
    end
    ret::T = d.edges[next_ind_l]
    if next_ind_l < next_ind_r
        ret += d.volumes[next_ind_l] * (d.probabilty_edges[next_ind_r] - r) * d.probabilty_inv_volumes[next_ind_l]
    end
    return ret
end


function Distributions.pdf(d::UvBinnedDist{T}, x::Real)::T where {T <: AbstractFloat}
    i::Int = StatsBase.binindex(d.h, x)
    return @inbounds d.h.weights[i]
end


function Distributions.logpdf(d::UvBinnedDist{T}, x::Real)::T where {T <: AbstractFloat}
    return log(pdf(d, x))
end


function Distributions.cdf(d::UvBinnedDist{T}, x::Real)::T where {T <: AbstractFloat}
    i::Int = StatsBase.binindex(d.h, x)
    p::T = @inbounds sum(d.h.weights[1:i-1] .* d.volumes[1:i-1])
    p += (x - d.edges[i]) * d.h.weights[i]
    return p
end


function Distributions.minimum(d::UvBinnedDist{T})::T where {T <: AbstractFloat}
    d.edges[1]
end

function Distributions.maximum(d::UvBinnedDist{T})::T where {T <: AbstractFloat}
    d.edges[end]
end


function Distributions.insupport(d::UvBinnedDist{T}, x::Real)::Bool where {T <: AbstractFloat}
    d.edges[1] <= x <= d.edges[end]
end


function Distributions.quantile(d::UvBinnedDist{T}, x::Real)::T where {T <: AbstractFloat}
    r::UnitRange{Int} = searchsorted(d.acc_prob, x)
    idx::Int = min(r.start, r.stop)
    p::T = d.acc_prob[ idx ]
    q::T = d.edges[idx]
    missing_p::T = x - p
    inv_weight::T = d.inv_weights[idx]
    if !isinf(inv_weight)
        q += missing_p * inv_weight
    end
    return min(q, maximum(d))
end


Base.eltype(d::UvBinnedDist{T}) where {T <: AbstractFloat}= T


Statistics.mean(d::UvBinnedDist) = d.μ

Statistics.var(d::UvBinnedDist) = d.var

Statistics.cov(d::UvBinnedDist) = d.cov
