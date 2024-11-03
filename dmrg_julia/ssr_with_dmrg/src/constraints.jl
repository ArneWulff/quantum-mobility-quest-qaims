"""
    ConstraintSettings

Defines the settings for implementing constraints in stacking sequence retrieval, supporting
both general input for constructing constraint MPOs and a user-friendly initialization for 
laminate engineers through specific keyword arguments for familiar constraints.

Fields:
  - `nn::Bool`: Whether a nearest-neighbor constraint is defined.
  - `knearest::Bool`: Whether a k-nearest neighbor constraint is defined.
  - `minimum::Bool`: Whether a minimum ply count constraint is defined.
  - `balanced::Bool`: Whether a balanced condition constraint is defined.

  - `nn_penalty::Union{Float64,Nothing}`: Penalty value for the nearest-neighbor constraint.
  - `nn_p_list::Union{Matrix{Float64},Nothing}`: Matrix for p-values in the nearest-neighbor constraint MPO.
  - `nn_q_list::Union{Matrix{Float64},Nothing}`: Matrix for q-values in the nearest-neighbor constraint MPO.

  - `knearest_penalty::Union{Float64,Nothing}`: Penalty value for the k-nearest neighbor constraint.
  - `knearest_plists::Union{Array{Float64,3},Nothing}`: Array of p-values for each ply in the k-nearest constraint MPO.

  - `minimum_penalty::Union{Float64,Nothing}`: Penalty for the minimum ply count constraint.
  - `minimum_t::Union{Vector{Tuple{Int,Int}},Nothing}`: List of (ply type, minimum count) tuples.

  - `balanced_penalty::Union{Float64,Nothing}`: Penalty for the balanced condition constraint.
  - `balanced_angles::Union{Vector{Tuple{Int,Int}},Nothing}`: Pairs of ply angles required to balance.

Usage:
  - **Direct Field Input**: Manually set each field for full control over constraint definitions.
  - **Keyword Constructor**: Use the `ConstraintSettings(laminate; kwargs...)` function to initialize 
    with engineering-standard constraints such as `disorientation`, `contiguity`, `percent`, and `balanced`.

"""
Base.@kwdef struct ConstraintSettings
    nn::Bool
    knearest::Bool
    minimum::Bool
    balanced::Bool

    nn_penalty::Union{Float64,Nothing} = nothing
    nn_p_list::Union{Matrix{Float64},Nothing} = nothing
    nn_q_list::Union{Matrix{Float64},Nothing} = nothing

    knearest_penalty::Union{Float64,Nothing} = nothing
    knearest_plists::Union{Array{Float64,3},Nothing} = nothing

    minimum_penalty::Union{Float64,Nothing} = nothing
    minimum_t::Union{Vector{Tuple{Int,Int}},Nothing} = nothing  # (t,t_min)

    balanced_penalty::Union{Float64,Nothing} = nothing
    balanced_angles::Union{Vector{Tuple{Int,Int}},Nothing} = nothing  # (s,t)
end

"""
    ConstraintSettings(laminate::Laminate; kwargs...)

Constructs a `ConstraintSettings` object based on familiar laminate engineering constraints 
by accepting keyword arguments for disorientation, contiguity, percent, and balanced rules.

Arguments:
  - `laminate::Laminate`: The laminate object representing ply properties.
  - `kwargs...`: Optional arguments to specify constraint parameters, including:
      - `disor_penalty::Float64`: Penalty for disorientation constraint.
      - `disor_dist::Float64`: Disorientation distance threshold for penalty.
      - `disor_angles::Vector{Float64}`: Ply angles involved in the disorientation constraint.
      - `contiguity_penalty::Float64`: Penalty for contiguity constraint.
      - `contiguity_dist::Int`: Number of plies allowed to have the same angle consecutively.
      - `perc_penalty::Float64`: Penalty for percent rule.
      - `perc_min::Float64`: Minimum percent threshold for each angle type.
      - `balanced_penalty::Float64`: Penalty for balanced constraint.
      - `balanced_angles::Vector{Tuple{Int,Int}}`: Pairs of ply angles required to balance.

Notes:
  - **Disorientation, Contiguity, Percent, Balanced**: When any parameter within a constraint group is specified, 
    the others must also be defined to ensure consistency. If a penalty parameter is omitted, it defaults to 1.0.
  - **Errors**: An error is raised if required parameters for a group are partially specified.
"""
function ConstraintSettings(laminate::Laminate; kwargs...)::ConstraintSettings
    args = Dict(kwargs)
    for arg_group in [
        (:disor_penalty, :disor_dist, :disor_angles),
        (:contiguity_penalty, :contiguity_dist),
        (:perc_penalty, :perc_min),
        (:balanced_penalty, :balanced_angles),
    ]
        values = [get(args, arg, nothing) for arg in arg_group]

        if all(v !== nothing for v in values[2:end]) && values[1] === nothing
            # Set the first argument to 1.0 if others are set but the first is not
            args[arg_group[1]] = 1.0
        end

        if any(v !== nothing for v in values) && any(v === nothing for v in values[2:end])
            error("Incomplete specification for constraint group $(arg_group). "
                  * "All arguments in this group must be set when any is provided, "
                  * "except for the first, which defaults to 1.0. Missing: "
                  * "$(filter(x -> args[x] === nothing, arg_group[2:end]))")
        end
    end

    nn,nn_p_list,nn_q_list = :disor_penalty in keys(args) ? (
        true, generate_disorientation_constraint_pq_list(
            args[:disor_angles],args[:disor_dist],args[:disor_penalty]
        )...
    ) : (false,nothing,nothing)

    knearest,knearest_plists = :contiguity_penalty in keys(args) ? (
        true, generate_plists_contiguity(
            num_angles(laminate),args[:contiguity_dist],args[:contiguity_penalty]
        )
    ) : (false,nothing)

    balanced = :balanced_penalty in keys(args)
    balanced_angles::Union{Vector{Tuple{Int,Int}},Tuple{Int,Int},Nothing} = get(args, :balanced_angles, nothing)
    if typeof(balanced_angles) == Tuple{Int,Int}
        balanced_angles = [balanced_angles]
    end

    if :perc_penalty in keys(args)
        minimum_constr = true
        balanced_angles_set = balanced_angles !== nothing ? Set(
            [b[j] for b in balanced_angles, j in 1:2]
        ) : Set{Int}()
        perc_vec = [a in balanced_angles_set ? args[:perc_min]/2 : args[:perc_min] for a in 1:num_angles(laminate)]
        minimum_t = [ia for ia in enumerate(ceil.(Int, perc_vec .* num_plies(laminate)))]
    else
        minimum_constr = false
        minimum_t = nothing
    end

    return ConstraintSettings(
        nn, knearest, minimum_constr, balanced,
        get(args, :disor_penalty, nothing), 
        nn_p_list, nn_q_list,
        get(args, :contiguity_penalty, nothing), 
        knearest_plists,
        get(args, :perc_penalty, nothing), 
        minimum_t,
        get(args, :balanced_penalty, nothing), 
        balanced_angles
    )
end

"""
    count_nn_constraint_violations(stack::Vector{Int}, constraints::ConstraintSettings)

Counts the number of nearest-neighbor constraint violations in the stacking sequence `stack`. 
This function evaluates each consecutive ply in `stack` using the nearest-neighbor constraint 
defined by `constraints.nn_p_list` and `constraints.nn_q_list`. A violation occurs whenever 
the dot product `p_list(s₁) * q_list(s₂)` is non-zero, where `s₁` and `s₂` are the states 
of consecutive plies.

Arguments:
  - `stack::Vector{Int}`: The stacking sequence represented as a vector of ply indices.
  - `constraints::ConstraintSettings`: The constraint settings, containing the `nn` flag 
    and matrices `nn_p_list` and `nn_q_list` if the nearest-neighbor constraint is active.

Returns:
  - `Int`: The number of nearest-neighbor constraint violations in `stack`.
"""
function count_nn_constraint_violations(stack::Vector{Int},constraints::ConstraintSettings)::Int
    if !constraints.nn
        return 0
    end

    return sum(0 .< sum(
        (constraints.nn_p_list[:,stack[1:end-1]] .* constraints.nn_q_list[:,stack[2:end]]),
        dims=1
    ))
end

"""
    count_knearest_constraint_violations(stack::Vector{Int}, constraints::ConstraintSettings)

Counts the number of k-nearest neighbor constraint violations in the stacking sequence `stack`.
The function evaluates groups of plies within `stack` according to the k-nearest neighbor 
constraint defined by `constraints.knearest_plists`. A violation occurs if the product of 
all vectors `p₁, p₂, ..., pₖ` associated with a group of `k` consecutive plies is non-zero.

Arguments:
  - `stack::Vector{Int}`: The stacking sequence represented as a vector of ply indices.
  - `constraints::ConstraintSettings`: The constraint settings, containing the `knearest` flag 
    and array `knearest_plists` if the k-nearest neighbor constraint is active.

Returns:
  - `Int`: The number of k-nearest neighbor constraint violations in `stack`.
"""
function count_knearest_constraint_violations(stack::Vector{Int},constraints::ConstraintSettings)::Int
    if !constraints.knearest
        return 0
    end

    # p_lists: (k,vec_length,num_angles)
    return sum(0 .< sum(reduce(.*, map(
        i -> constraints.knearest_plists[
            i, :, stack[i:length(stack)-size(constraints.knearest_plists,1)+i]
        ], 1:size(constraints.knearest_plists,1)
    )), dims=2))
end

"""
    count_balanced_constraint_violations(stack::Vector{Int}, constraints::ConstraintSettings)

Counts the number of balanced constraint violations in the stacking sequence `stack`. 
For each specified pair of ply angles `(b₁, b₂)` in `constraints.balanced_angles`, the 
function checks if the occurrences of `b₁` and `b₂` are equal. Each imbalance between 
`b₁` and `b₂` counts as one violation.

Arguments:
  - `stack::Vector{Int}`: The stacking sequence represented as a vector of ply indices.
  - `constraints::ConstraintSettings`: The constraint settings, containing the `balanced` flag 
    and pairs `balanced_angles` if the balanced constraint is active.

Returns:
  - `Int`: The number of balanced constraint violations in `stack`.
"""
function count_balanced_constraint_violations(stack::Vector{Int},constraints::ConstraintSettings)::Int
    if !constraints.balanced
        return 0
    end

    return sum(
        abs(sum(stack .== b1) - sum(stack .== b2)) 
        for (b1,b2) in constraints.balanced_angles
    )
end

"""
    count_minimum_constraint_violations(stack::Vector{Int}, constraints::ConstraintSettings)

Counts the number of minimum ply count constraint violations in the stacking sequence `stack`.
For each ply angle `t` specified in `constraints.minimum_t`, the function checks if the 
number of occurrences of `t` meets the required minimum count. Each shortfall counts as one 
violation.

Arguments:
  - `stack::Vector{Int}`: The stacking sequence represented as a vector of ply indices.
  - `constraints::ConstraintSettings`: The constraint settings, containing the `minimum` flag 
    and list `minimum_t` if the minimum constraint is active.

Returns:
  - `Int`: The number of minimum ply count constraint violations in `stack`.
"""
function count_minimum_constraint_violations(stack::Vector{Int},constraints::ConstraintSettings)::Int
    if !constraints.minimum
        return 0
    end

    return sum(
        max(t_min-sum(stack .== t),0)
        for (t,t_min) in constraints.minimum_t
    )
end

"""
    count_constraint_violations(stack::Vector{Int}, constraints::ConstraintSettings)

Returns a dictionary containing counts of constraint violations for all active constraints 
in the `ConstraintSettings` object. The dictionary keys are the names of the constraints 
(`"nn"`, `"knearest"`, `"minimum"`, and `"balanced"`), and the values are the respective 
violation counts as computed by the individual constraint violation functions.

Arguments:
  - `stack::Vector{Int}`: The stacking sequence represented as a vector of ply indices.
  - `constraints::ConstraintSettings`: The constraint settings that specify which constraints 
    are active and their parameters.

Returns:
  - `Dict{String,Int}`: A dictionary where each key represents a constraint type and each 
    value is the count of violations for that constraint in `stack`.
"""
function count_constraint_violations(stack::Vector{Int},constraints::ConstraintSettings)::Dict{String,Int}
    return Dict(
        string(key) => func(stack,constraints) for (key,func) in [
            (:nn, count_nn_constraint_violations),
            (:knearest, count_knearest_constraint_violations),
            (:minimum, count_minimum_constraint_violations),
            (:balanced, count_balanced_constraint_violations)
        ] if getfield(constraints, key)
    )
end