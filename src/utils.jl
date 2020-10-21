# This file is a part of EmpiricalDistributions.jl, licensed under the MIT License (MIT).


_bin_left(edge::AbstractVector{<:Real}) = view(edge, firstindex(edge):lastindex(edge)-1)
_bin_right(edge::AbstractVector{<:Real}) = view(edge, firstindex(edge)+1:lastindex(edge))
_bin_centers(edge::AbstractVector{<:Real}) = (_bin_left(edge) .+ _bin_right(edge)) ./ 2

_bin_widths(edge::AbstractVector{<:Real}) = _bin_right(edge) .- _bin_left(edge)

# Ranges need special treatment, since subtraction of integer ranges results in "step cannot be zero" error
function _bin_widths(edge::AbstractRange{<:Real})
    ref = step(edge)
    T = typeof(ref)
    stp = zero(T)
    n = length(eachindex(edge)) - 1
    StepRangeLen(ref, stp, n)
end


function _ratio(a::Real, b::Real)
    R = float(promote_type(typeof(a), typeof(b)))
    a == b ?  one(R) : convert(R, a / b)
end


function _linear_interpol(x_lo::Real, x_hi::Real, y_lo::Real, y_hi::Real, x::Real)
    T = promote_type(typeof(y_lo), typeof(y_hi), typeof(x))
    w_hi = T(_ratio(x - x_lo, x_hi - x_lo))
    w_lo = one(w_hi) - w_hi
    y = y_lo * w_lo + y_hi * w_hi
    @assert y_lo <= y <= y_hi
    y
end


function _linear_interpol(X::AbstractVector{<:Real}, Y::AbstractVector{<:Real}, x::Real)
    i_lo, i_hi = _find_idxs_lohi(X, x)
    x_lo, x_hi = X[i_lo], X[i_hi]
    y_lo, y_hi = Y[i_lo], Y[i_hi]
    _linear_interpol(x_lo, x_hi, y_lo, y_hi, x)
end



function _find_bin(edge::AbstractVector{<:Real}, closed_left::Bool, x::Real)
    if closed_left == false
        searchsortedfirst(edge, x) - 1
    else
        searchsortedlast(edge, x)
    end
end

function _find_bin(edges::NTuple{N,AbstractVector{<:Real}}, closed_left::Bool, x::NTuple{N,Real}) where N
    map(_find_bin, edges, map(_ -> closed_left, x), x)
end


function _find_idxs_lohi(X::AbstractVector{<:Real}, x::Real)
    x_min, x_max = first(X), last(X)
    x_min <= x <= x_max || throw(DomainError(x, "require value between x_min and x_max"))
    idxs = searchsorted(X, x)
    i1, i2 = first(idxs), last(idxs)
    i1, i2 = first(idxs), last(idxs)
    from, to = min(i1, i2), max(i1, i2)
    @assert checkbounds(Bool, X, from:to)
    (from, to)
end


function _mean(h::StatsBase.Histogram{<:Real, N}) where {N}
    @assert h.isdensity # Implementation requires normalized histogram

    Y = h.weights
    X = _bin_centers.(h.edges)
    W = _bin_widths.(h.edges)
    
    mean_est = mapreduce(
        idx -> map(getindex, X, idx.I) .* Y[idx] .* prod(map(getindex, W, idx.I)),
        (a, b) -> a .+ b,
        CartesianIndices(Y)
    ) ./ sum(idx -> Y[idx] .* prod(map(getindex, W, idx.I)), CartesianIndices(Y))
end


_findmaxidx_tuple_or_int(A::AbstractVector{<:Real}) = findmax(A)[2]
_findmaxidx_tuple_or_int(A::AbstractArray{<:Real}) = findmax(A)[2].I

function _mode(h::StatsBase.Histogram)
    @assert h.isdensity # Implementation requires normalized histogram

    maxidx = _findmaxidx_tuple_or_int(h.weights)
    mode_corner1 = map(getindex, h.edges, maxidx)
    mode_corner2 = map(getindex, h.edges, maxidx .+ 1)
    (mode_corner1 .+ mode_corner2) ./ 2
end


function _var(h::StatsBase.Histogram{<:Real,N}, mean_est::NTuple{N,Real}) where {N}
    @assert h.isdensity # Implementation requires normalized histogram

    Y = h.weights
    X = _bin_centers.(h.edges)
    W = _bin_widths.(h.edges)
    
    mean_est = mapreduce(
        idx -> (map(getindex, X, idx.I) .- mean_est).^2 .* Y[idx] .* prod(map(getindex, W, idx.I)),
        (a, b) -> a .+ b,
        CartesianIndices(Y)
    ) ./ sum(idx -> Y[idx] .* prod(map(getindex, W, idx.I)), CartesianIndices(Y))
end


function _cov(h::StatsBase.Histogram{<:Real,N}, mean_est::NTuple{N,Real}) where {N}
    @assert h.isdensity # Implementation requires normalized histogram

    Y = h.weights
    X = _bin_centers.(h.edges)
    W = _bin_widths.(h.edges)
    
    dim_idxs = 1:N

    [
        mapreduce(
            idx -> (X[i][idx.I[i]] - mean_est[i]) * (X[j][idx.I[j]] - mean_est[j]) .* Y[idx] .* prod(map(getindex, W, idx.I)),
            (a, b) -> a .+ b,
            CartesianIndices(Y)
        ) ./ sum(idx -> Y[idx] .* prod(map(getindex, W, idx.I)), CartesianIndices(Y))
        for i in dim_idxs, j in dim_idxs
    ]
end


_rand_uniform(rng::AbstractRNG, T::Type{<:Real}, lo::Real, hi::Real) = fma(hi - lo, rand(rng, T), lo)


# For tests of adapt()
struct _AdaptToArray end
Adapt.adapt_storage(::Val{_AdaptToArray}, xs::AbstractArray) = convert(Array, xs)
