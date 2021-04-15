export UvDiscreteDist

"""
    UvDiscreteDist(data::Vector{T} where T <: Real)::Distribution{Univariate,Discrete}

Create a discrete empirical distribution based on `data`.
"""
function UvDiscreteDist(data::Vector{T} where T <: Real)::Distribution{Univariate,Discrete}
    data = float.(data) #convert any int to float (to allow for integration if desired)
    sort!(data) #sort the observations
    empirical_cdf = ecdf(data) #create empirical cdf
    data_clean = unique(data) #remove duplicates to avoid allunique error
    cdf_data = empirical_cdf.(data_clean) #apply ecdf to data
    pmf_data = vcat(cdf_data[1],diff(cdf_data)) #create pmf from the cdf
    DiscreteNonParametric(data_clean,pmf_data) #define distribution
end
