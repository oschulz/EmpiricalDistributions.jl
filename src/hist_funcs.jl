# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).


function _pdf(h::Histogram{T,N}, xs::NTuple{N,Real}) where {T,N}
    @assert h.isdensity # Implementation requires normalized histogram

    idx = StatsBase.binindex(h, xs)
    r::T = zero(T)
    if checkbounds(Bool, h.weights, idx...)
        @inbounds r = h.weights[idx...]
    end
    r
end


function _mean(h::StatsBase.Histogram{<:Real, N}; T::DataType = Float64) where {N}
    @assert !h.isdensity # Implementation currently assumes non-normalized histogram

    s_inv::T = inv(sum(h.weights))
    m::Vector{T} = zeros(T, N)
    mps = StatsBase.midpoints.(h.edges)
    cart_inds = CartesianIndices(h.weights)
    for i in cart_inds
        for idim in 1:N
            m[idim] += s_inv * mps[idim][i[idim]] * h.weights[i]
        end
    end
    return m
end


_findmaxidx_tuple_or_int(A::AbstractVector{<:Real}) = findmax(A)[2]
_findmaxidx_tuple_or_int(A::AbstractArray{<:Real}) = findmax(A)[2].I

function _mode(h::StatsBase.Histogram; T::DataType = Float64)
    @assert h.isdensity # Implementation requires normalized histogram

    maxidx = _findmaxidx_tuple_or_int(h.weights)
    mode_corner1 = map(getindex, h.edges, maxidx)
    mode_corner2 = map(getindex, h.edges, maxidx .+ 1)
    cov_est = T[(mode_corner1 .+ mode_corner2) ./ 2...]
end


function _var(h::StatsBase.Histogram{<:Real, N}; T::DataType = Float64, mean = StatsBase.mean(h, T = T), ) where {N}
    @assert !h.isdensity # Implementation currently assumes non-normalized histogram

    s_inv::T = inv(sum(h.weights))
    v::Vector{T} = zeros(T, N)
    mps = StatsBase.midpoints.(h.edges)
    cart_inds = CartesianIndices(h.weights)
    for i in cart_inds
        for idim in 1:N
            v[idim] += s_inv * (mps[idim][i[idim]] - mean[idim])^2 * h.weights[i]
        end
    end
    return v
end


function _cov(h::StatsBase.Histogram{<:Real, N}; T::DataType = Float64, mean = StatsBase.mean(h, T = T)) where {N}
    @assert !h.isdensity # Implementation currently assumes non-normalized histogram

    s_inv::T = inv(sum(h.weights))
    c::Matrix{T} = zeros(T, N, N)
    mps = StatsBase.midpoints.(h.edges)
    cart_inds = CartesianIndices(h.weights)
    for i in cart_inds
        for idim in 1:N
            for jdim in 1:N
                c[idim, jdim] += s_inv * (mps[idim][i[idim]] - mean[idim]) * (mps[jdim][i[jdim]] - mean[jdim]) * h.weights[i]
            end
        end
    end
    return c
end
