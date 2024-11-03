"""
    build_mpo_constr_minimum_middle(num_angles, t, t_min, lefti, righti, upi, downi)

Constructs the middle tensor in the MPO chain for the minimum constraint,
tracking occurrences of a specific ply angle.

# Arguments
- `num_angles::Int`: Total number of ply angle states.
- `t::Int`: Index of the ply angle state to track.
- `t_min::Int`: Minimum required number of occurrences for angle `t`.
- `lefti::Index`, `righti::Index`: Bond indices connecting to adjacent tensors.
- `upi::Index`, `downi::Index`: Physical indices representing ply angle states.

# Returns
- `ITensor`: Middle tensor for the minimum constraint MPO.
"""
function build_mpo_constr_minimum_middle(num_angles::Int,t::Int,t_min::Int,
    lefti::Index, righti::Index, upi::Index, downi::Index
)

    bond_dim = t_min + 1

    arr = reshape(
        diagm(ones(Float64,bond_dim)),bond_dim,bond_dim,1,1
    ) .* reshape(
        diagm(ones(Float64,num_angles)),1,1,num_angles,num_angles
    )

    arr[CartesianIndex.(2:bond_dim,2:bond_dim,t,t)] = zeros(Float64,bond_dim-1)
    arr[CartesianIndex.(1:(bond_dim-1),2:bond_dim,t,t)] = ones(Float64,bond_dim-1)

    return ITensor(arr,lefti,righti,upi,downi)
end

"""
    build_mpo_constr_minimum_begin(num_angles, t, t_min, γ, righti, upi, downi)

Constructs the beginning tensor in the MPO chain for the minimum constraint, initializing the count.

# Arguments
- `num_angles::Int`: Total number of ply angle states.
- `t::Int`: Index of the ply angle state to track.
- `t_min::Int`: Minimum required number of occurrences for angle `t`.
- `γ::Float64`: Penalty factor for constraint violation.
- `righti::Index`: Bond index connecting to the next tensor.
- `upi::Index`, `downi::Index`: Physical indices representing ply angle states.

# Returns
- `ITensor`: Beginning tensor for the minimum constraint MPO.
"""
function build_mpo_constr_minimum_begin(
    num_angles::Int,t::Int,t_min::Int,γ::Float64,
    righti::Index, upi::Index, downi::Index
)
    bond_dim = t_min + 1

    base_vec = zeros(Float64,bond_dim)
    base_vec[1] = γ

    arr = reshape(
        base_vec,bond_dim,1,1
    ) .* reshape(
        diagm(ones(Float64,num_angles)),1,num_angles,num_angles
    )

    arr[2,t,t] = γ
    return ITensor(arr,righti,upi,downi)
end

"""
    build_mpo_constr_minimum_end(num_angles, t, t_min, lefti, upi, downi)

Constructs the ending tensor in the MPO chain for the minimum constraint, finalizing the count and applying penalties if the minimum count is not met.

# Arguments
- `num_angles::Int`: Total number of ply angle states.
- `t::Int`: Index of the ply angle state to track.
- `t_min::Int`: Minimum required number of occurrences for angle `t`.
- `lefti::Index`: Bond index connecting from the previous tensor.
- `upi::Index`, `downi::Index`: Physical indices representing ply angle states.

# Returns
- `ITensor`: Ending tensor for the minimum constraint MPO.
"""
function build_mpo_constr_minimum_end(
    num_angles::Int,t::Int,t_min::Int,
    lefti::Index, upi::Index, downi::Index
)
    bond_dim = t_min + 1

    base_vec = zeros(Float64,bond_dim)
    base_vec[1] = Float64(t_min)
    base_vec[2:bond_dim] .= -1.

    arr = reshape(
        base_vec,bond_dim,1,1
    ) .* reshape(
        diagm(ones(Float64,num_angles)),1,num_angles,num_angles
    )

    arr[1,t,t] -= 1.
    arr[bond_dim,t,t] = 0.

    return ITensor(arr,lefti,upi,downi)
end

"""
    build_mpo_constr_minimum(sites, num_states, t, t_min, γ)

Constructs the Matrix Product Operator (MPO) for enforcing a minimum constraint on ply angle
occurrences, such as the 10% rule, across the laminate. This constraint applies a penalty `γ`
if the number of plies with a specific angle (given by `t`) falls below the minimum threshold `t_min`.

# Arguments
- `sites::Vector{<:Index}`: List of site indices for each ply in the laminate,
    representing the ply angles.
- `num_states::Int`: Total number of possible ply angle states.
- `t::Int`: Index of the ply angle state that the constraint applies to.
- `t_min::Int`: Minimum required number of occurrences for angle `t`.
- `γ::Float64`: Penalty factor for violating the minimum constraint.

# Returns
- `MPO`: The MPO representation of the minimum constraint Hamiltonian.

# Details
This MPO checks the count of a specified ply angle `t` across consecutive plies. The bond dimension increases according to `t_min`, ensuring that each ply with angle `t` contributes to the count towards satisfying the threshold. Penalties are applied if the minimum count is not reached, thus enforcing the minimum occurrence requirement.

This constraint is commonly used in laminate design, where structural requirements often impose a minimum count on certain ply angles for balanced properties.
"""
function build_mpo_constr_minimum(
    sites::Vector{<:Index},
    num_states::Int,t::Int,t_min::Int,γ::Float64
)
    num_sites = length(sites)
    @assert num_sites > 1
    
    bond_dim = t_min+1
    links = [Index(bond_dim, "Link,l=$n") for n ∈ 1:(num_sites-1)]
    Hmpo = MPO(sites)
  
    # first site
    Hmpo[1] = build_mpo_constr_minimum_begin(num_states,t,t_min,γ,links[1],sites[1],sites[1]')
    
    # last site
    Hmpo[end] = build_mpo_constr_minimum_end(num_states,t,t_min,links[end],sites[end],sites[end]')
  
    if num_sites == 2
      return Hmpo
    end
  
    # second site
    Hmpo[2] = build_mpo_constr_minimum_middle(
        num_states,t,t_min,links[1],links[2],sites[2],sites[2]'
    )
  
    # fill up other sites with copies
    for n ∈ 3:(num_sites-1)
      Hmpo[n] = replaceinds!(copy(Hmpo[2]),[links[1],links[2],sites[2],sites[2]'],[links[n-1],links[n],sites[n],sites[n]'])
    end
  
    return Hmpo
  end