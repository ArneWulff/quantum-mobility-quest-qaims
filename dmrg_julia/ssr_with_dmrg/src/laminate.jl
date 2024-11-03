"""
    struct Laminate

The `Laminate` struct represents composite laminate properties used in stacking sequence retrieval. 

Fields:
    - `weights::Matrix{Float64}`: A matrix (num_weights x num_plies) representing ply weighting factors.
    - `funcs::Matrix{Float64}`: A matrix (num_angles x num_funcs) of angle functions, based on trigonometric function values.

Example:
    lam = Laminate(weights_matrix, funcs_matrix)
"""
struct Laminate
    weights::Matrix{Float64}  # size (num_weights, num_plies)
    funcs::Matrix{Float64}    # size (num_angles, num_funcs)
end

"""
    Laminate(num_plies, angles; symmetric=true, deg=true, weight_types=nothing, func_types=nothing)

Constructs a `Laminate` instance with generated weight and function matrices.

Arguments:
    - `num_plies::Int`: Number of plies in the laminate.
    - `angles::Vector{<:Real}`: Vector of ply angles.
    - `symmetric::Bool=true`: If true, creates a symmetric weight matrix.
    - `deg::Bool=true`: If true, interprets `angles` in degrees, otherwise in radians.
    - `weight_types`: Types of ply weights, e.g., "ABD".
    - `func_types`: Types of trigonometric functions for angle calculations, e.g., ["cos2", "sin2", "cos4", "sin4"].

Returns:
    - `Laminate`: An initialized `Laminate` struct with generated weight and function matrices.
"""
function Laminate(num_plies::Int,angles::Vector{<:Real},
    symmetric::Bool=true,deg::Bool=true,
    weight_types::Union{String,AbstractVector{<:Union{Char,String}},Tuple{Vararg{Union{Char,String}}},Nothing}=nothing,
    func_types::Union{Vector{String},Tuple{Vararg{String}},Nothing}=nothing
)
    weights = weight_types === nothing ? generate_weights(num_plies,symmetric) : generate_weights(num_plies,symmetric,weight_types)
    funcs = func_types === nothing ? generate_funcs(angles,deg) : generate_funcs(angles,deg,func_types)

    return Laminate(weights,funcs)::Laminate
end

"""
    num_plies(lam::Laminate)

Returns the number of plies in the `Laminate`.
"""
function num_plies(lam::Laminate)::Int
    return size(lam.weights)[2]
end

"""
    num_angles(lam::Laminate)

Returns the number of angles in the `Laminate`.
"""
function num_angles(lam::Laminate)::Int
    return size(lam.funcs)[1]
end

"""
    num_weights(lam::Laminate)

Returns the number of weights (e.g., 3 for (A, B, D)) in the `Laminate`.
"""
function num_weights(lam::Laminate)::Int
    return size(lam.weights)[1]
end

"""
    num_funcs(lam::Laminate)

Returns the number of angle functions (e.g., 4 for cos(2x), sin(2x),
cos(4x), sin(4x)) in the `Laminate`.
"""
function num_funcs(lam::Laminate)::Int
    return size(lam.funcs)[2]
end

"""
    parameters(lam::Laminate, stack::Vector{T}) where T<:Integer

Calculates the lamination parameters based on the stacking sequence.

Arguments:
    - `lam::Laminate`: The `Laminate` instance.
    - `stack::Vector{T}`: Vector representing the stacking sequence with 
        entries corresponding to ply angles.

Returns:
    - `Matrix{Float64}`: Lamination parameters as a matrix (num_weights x num_funcs)
"""
function parameters(lam::Laminate,stack::Vector{T})::Matrix{Float64} where T<:Integer
    # stack: size (num_plies,), entries in {1,2,...,num_angles}
    return lam.weights * lam.funcs[stack, :]  # size
end

"""
    generate_weights(
        num_plies::Integer, symmetric::Bool=true, weight_types="ABD"
    )

Generates a weight matrix based on ply configuration.

Arguments:
    - `num_plies::Integer`: Number of plies.
    - `symmetric::Bool=true`: If true, creates weights for a symmetric laminate.
    - `weight_types::String`: String specifying weight types, e.g., "ABD" for different ply types.

Returns:
    - `Matrix{Float64}`: A (num_weights x num_plies) weight matrix.
"""
function generate_weights(
    num_plies::Integer,symmetric::Bool,weight_types::String
)::Matrix{Float64}
    boundaries_ref = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    num_weights = length(weight_types)

    # column-wise fill for better performance even with tranpose (simplified test: 383 ns vs. 521 ns, same allocation 4.06 KiB)
    weights = Matrix{Float64}(undef,num_plies,length(weight_types))  

    for (idx,w) in enumerate(weight_types)
        if w == 'A'
            weights[:,idx] .= 1/num_plies
        else
            # calculate boundaries
            if boundaries_ref[] === nothing
                # in test fastest from 
                # ∘ range( [0 ∧ -0.5], stop = [1 ∧ 0.5], length=num_plies+1) [.- 0.5]
                # ∘ (0:num_plies)./num_plies [.- 0.5]
                # ∘ [0 ∧ (-0.5)]:(1/num_plies):[1 ∧ 0.5]
                boundaries_ref[] = symmetric ? range(0, stop=1, length=num_plies+1) : range(0, stop=1, length=num_plies+1) .- 0.5 # collect() ?
            end
            pot,fac = w=='B' ? (2,2) : (3,4)   # else is "D"
            boundaries_pot = symmetric ? boundaries_ref[].^pot : (boundaries_ref[].^pot) .* fac
            weights[:,idx] = boundaries_pot[2:end] - boundaries_pot[1:(end-1)]
        end
    end

    return transpose(weights)
end

function generate_weights(num_plies::Integer,symmetric::Bool=true)
    weight_types = symmetric ? "AD" : "ABD"
    return generate_weights(num_plies,symmetric,weight_types)
end

function generate_weights(num_plies::Integer,symmetric::Bool,weight_types::Union{AbstractVector{<:Union{Char,String}},Tuple{Vararg{Union{Char,String}}}})
    return generate_weights(num_plies,symmetric,join(map(String,weight_types)))
end

cosdeg2 = (x) -> cospi(x / 90)
sindeg2 = (x) -> sinpi(x / 90)
cosdeg4 = (x) -> cospi(x / 45)
sindeg4 = (x) -> sinpi(x / 45)
cosrad2 = (x) -> cos(2 * x)
sinrad2 = (x) -> sin(2 * x)
cosrad4 = (x) -> cos(4 * x)
sinrad4 = (x) -> sin(4 * x)

const FUNC_OPTIONS_DEG = Dict(
    "cos2" => cosdeg2,
    "c2" => cosdeg2,
    "sin2" => sindeg2,
    "s2" => sindeg2,
    "cos4" => cosdeg4,
    "c4" => cosdeg4,
    "sin4" => sindeg4,
    "s4" => sindeg4
)

const FUNC_OPTIONS_RAD = Dict(
    "cos2" => cosrad2,
    "c2" => cosrad2,
    "sin2" => sinrad2,
    "s2" => sinrad2,
    "cos4" => cosrad4,
    "c4" => cosrad4,
    "sin4" => sinrad4,
    "s4" => sinrad4
)

"""
    generate_funcs(
        angles::Vector{<:Real}, deg::Bool=true, func_types=["cos2", "sin2", "cos4", "sin4"]
    )

Generates a function matrix using specified trigonometric functions for each ply angle.

Arguments:
    - `angles::Vector{<:Real}`: Vector of angles, in degrees or radians.
    - `deg::Bool=true`: If true, interprets `angles` in degrees.
    - `func_types::Vector{String}`: Vector specifying the trigonometric functions to use.

Returns:
    - `Matrix{Float64}`: A (num_angles x num_funcs) matrix of function values.
"""
function generate_funcs(
    angles::Vector{<:Real},deg::Bool,func_types::Vector{String}
)::Matrix{Float64}
    funcs = Matrix{Float64}(undef,length(angles),length(func_types))

    func_dict = deg ? FUNC_OPTIONS_DEG : FUNC_OPTIONS_RAD

    for (idx,f) in enumerate(func_types)
        funcs[:,idx] = func_dict[f].(angles)
    end

    return funcs
end

function generate_funcs(angles::Vector{<:Real},deg::Bool,func_types::Tuple{Vararg{String}})
    return generate_funcs(angles,deg,collect(func_types))
end

function generate_funcs(angles::Vector{<:Real},deg::Bool=true)
    return generate_funcs(angles,deg,["cos2","sin2","cos4","sin4"])
end

function generate_funcs(angles::Vector{<:Real},func_types::Union{Vector{String},Tuple{Vararg{String}}})
    return generate_funcs(angles,true,func_types)
end
