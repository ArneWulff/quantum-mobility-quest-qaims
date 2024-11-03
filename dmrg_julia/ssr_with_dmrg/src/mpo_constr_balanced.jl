"""
    build_mpo_constr_balanced_middle(num_angles, s, t, γ, lefti, righti, upi, downi)

Constructs the middle tensor in the MPO chain for the balanced constraint.

# Arguments
- `num_angles::Int`: Number of ply angle states.
- `s::Int`: Index of the first angle state to be balanced.
- `t::Int`: Index of the second angle state to be balanced.
- `γ::Float64`: Penalty applied for imbalance.
- `lefti::Index`, `righti::Index`: Bond indices connecting to adjacent tensors.
- `upi::Index`, `downi::Index`: Physical indices representing ply angle states.

# Returns
- `ITensor`: Middle tensor of the balanced constraint MPO.
"""
function build_mpo_constr_balanced_middle(num_angles::Int,s::Int,t::Int,γ::Float64,
    lefti::Index, righti::Index, upi::Index, downi::Index
)::ITensor
    val = sqrt(2)*sqrt(γ)

    arr = reshape(
        diagm(ones(Float64,3)),3,3,1,1
    ) .* reshape(
        diagm(ones(Float64,num_angles)),1,1,num_angles,num_angles
    )

    arr[1,2,s,s] = arr[2,3,s,s] = val
    arr[1,2,t,t] = arr[2,3,t,t] = -val
    arr[1,3,s,s] = arr[1,3,t,t] = γ

    return ITensor(arr,lefti,righti,upi,downi)
end

"""
    build_mpo_constr_balanced_begin(num_angles, s, t, γ, righti, upi, downi)

Constructs the beginning tensor in the MPO chain for the balanced constraint.

# Arguments
- `num_angles::Int`: Number of ply angle states.
- `s::Int`: Index of the first angle state to be balanced.
- `t::Int`: Index of the second angle state to be balanced.
- `γ::Float64`: Penalty applied for imbalance.
- `righti::Index`: Bond index connecting to the next tensor.
- `upi::Index`, `downi::Index`: Physical indices representing ply angle states.

# Returns
- `ITensor`: Beginning tensor of the balanced constraint MPO.
"""
function build_mpo_constr_balanced_begin(num_angles::Int,s::Int,t::Int,γ::Float64,
    righti::Index, upi::Index, downi::Index
)::ITensor
    val = sqrt(2)*sqrt(γ)

    arr = reshape(
        [1.,0.,0.],3,1,1
    ) .* reshape(
        diagm(ones(Float64,num_angles)),1,num_angles,num_angles
    )

    arr[2,s,s] = val
    arr[2,t,t] = -val
    arr[3,s,s] = arr[3,t,t] = γ

    return ITensor(arr,righti,upi,downi)
end

"""
    build_mpo_constr_balanced_end(num_angles, s, t, γ, lefti, upi, downi)

Constructs the ending tensor in the MPO chain for the balanced constraint.

# Arguments
- `num_angles::Int`: Number of ply angle states.
- `s::Int`: Index of the first angle state to be balanced.
- `t::Int`: Index of the second angle state to be balanced.
- `γ::Float64`: Penalty applied for imbalance.
- `lefti::Index`: Bond index connecting from the previous tensor.
- `upi::Index`, `downi::Index`: Physical indices representing ply angle states.

# Returns
- `ITensor`: Ending tensor of the balanced constraint MPO.
"""
function build_mpo_constr_balanced_end(num_angles::Int,s::Int,t::Int,γ::Float64,
    lefti::Index, upi::Index, downi::Index
)::ITensor
    val = sqrt(2)*sqrt(γ)

    arr = reshape(
        [0.,0.,1.],3,1,1
    ) .* reshape(
        diagm(ones(Float64,num_angles)),1,num_angles,num_angles
    )

    arr[2,s,s] = val
    arr[2,t,t] = -val
    arr[1,s,s] = arr[1,t,t] = γ
    
    return ITensor(arr,lefti,upi,downi)
end


"""
    build_mpo_constr_balanced(sites, num_states, s, t, γ)

Constructs the complete Matrix Product Operator (MPO) for the balanced constraint, 
which enforces a balance between specific ply angles in the laminate. This constraint 
is designed to apply a penalty `γ` when the number of plies at angles `s` and `t` differ. 

This MPO operates on the given set of `sites` by checking for imbalances at each ply, 
thus ensuring that any imbalance between angles `s` and `t` across all sites contributes 
to the constraint penalty.

# Arguments
- `sites::Vector{<:Index}`: List of site indices for the MPO, defining 
  the ply angle indices in the laminate.
- `num_states::Int`: Number of possible ply angle states; typically corresponds to the 
  distinct angles used in the laminate.
- `s::Int`: Index of the first ply angle state to be balanced.
- `t::Int`: Index of the second ply angle state to be balanced.
- `γ::Float64`: Penalty factor applied when the balance between `s` and `t` is not met.

# Returns
- `MPO`: The MPO representation of the balanced constraint Hamiltonian.
"""
function build_mpo_constr_balanced(sites::Vector{<:Index},
    num_states::Int,s::Int,t::Int,γ::Float64
)::MPO
    num_sites = length(sites)
    @assert num_sites > 1
    links = [Index(3, "Link,l=$n") for n ∈ 1:(num_sites-1)]
    Hmpo = MPO(sites)
  
    # first site
    Hmpo[1] = build_mpo_constr_balanced_begin(
        num_states,s,t,γ,links[1],sites[1],sites[1]'
    )
    
    # last site
    Hmpo[end] = build_mpo_constr_balanced_end(
        num_states,s,t,γ,links[end],sites[end],sites[end]'
    )
  
    if num_sites == 2
      return Hmpo
    end
  
    # second site
    Hmpo[2] = build_mpo_constr_balanced_middle(
        num_states,s,t,γ,links[1],links[2],sites[2],sites[2]'
    )
  
    # fill up other sites with copies
    for n ∈ 3:(num_sites-1)
      Hmpo[n] = replaceinds!(copy(Hmpo[2]),[links[1],links[2],sites[2],sites[2]'],[links[n-1],links[n],sites[n],sites[n]'])
    end
  
    return Hmpo
  end