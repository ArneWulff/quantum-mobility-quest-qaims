"""
    build_mpo_constr_nn_middle(p_list, q_list, lefti, righti, upi, downi)

Creates the middle tensor of the MPO for a nearest-neighbor constraint 
on ply angles between adjacent plies. This function allows encoding general 
constraints through `p_list` and `q_list` matrices, where `p_list[:, s1] ⋅ q_list[:, s2]` 
returns a penalty value if the constraint between angles `s1` and `s2` is violated.

# Arguments
- `p_list::Matrix{<:Union{Int,Float64}}`: Matrix of shape `(vector length, num_angles)` 
  defining the constraint check for each ply state. The `p_list[:, s1]` vector is 
  checked against `q_list[:, s2]` of the adjacent ply.
- `q_list::Matrix{<:Union{Int,Float64}}`: Matrix of shape `(vector length, num_angles)` 
  specifying the penalty values for violating constraints between ply states.
- `lefti::Index`, `righti::Index`: Bond dimensions for the MPO links.
- `upi::Index`, `downi::Index`: Physical indices representing ply angle states.

# Returns
- `ITensor`: Tensor encoding the middle site of the MPO constraint.
"""
function build_mpo_constr_nn_middle(p_list::Matrix{<:Union{Int,Float64}},
    q_list::Matrix{<:Union{Int,Float64}},
    lefti::Index, righti::Index, upi::Index, downi::Index
)::ITensor
    num_angles = size(p_list,2)
    bond_dim = size(p_list,1) + 2

    base_arr = zeros(Float64,bond_dim,bond_dim)
    base_arr[1,1] = base_arr[bond_dim,bond_dim] = 1.

    arr = reshape(
        base_arr,bond_dim,bond_dim,1,1
    ) .* reshape(
        diagm(ones(Float64,num_angles)),1,1,num_angles,num_angles
    )

    arr[1,2:(bond_dim-1),1:num_angles,1:num_angles] = reshape(p_list,bond_dim-2,num_angles,1) .* reshape(diagm(ones(Float64,num_angles)),1,num_angles,num_angles)
    arr[2:(bond_dim-1),bond_dim,1:num_angles,1:num_angles] = reshape(q_list,bond_dim-2,num_angles,1) .* reshape(diagm(ones(Float64,num_angles)),1,num_angles,num_angles)

    return ITensor(arr,lefti,righti,upi,downi)
end

"""
    build_mpo_constr_nn_begin(p_list, righti, upi, downi)

Creates the beginning tensor of the MPO for nearest-neighbor constraints, 
initializing the MPO chain for enforcing the ply angle restrictions.

# Arguments
- `p_list::Matrix{<:Union{Int,Float64}}`: Matrix defining constraint check vectors 
  for each ply state.
- `righti::Index`: Bond dimension for the link to the next tensor.
- `upi::Index`, `downi::Index`: Physical indices representing ply angle states.

# Returns
- `ITensor`: Tensor encoding the start of the MPO constraint.
"""
function build_mpo_constr_nn_begin(
    p_list::Matrix{<:Union{Int,Float64}},
    righti::Index, upi::Index, downi::Index
)::ITensor
    num_angles = size(p_list,2)
    bond_dim = size(p_list,1) + 2

    base_arr = zeros(Float64,bond_dim)
    base_arr[1] = 1.

    arr = reshape(
        base_arr,bond_dim,1,1
    ) .* reshape(
        diagm(ones(Float64,num_angles)),1,num_angles,num_angles
    )

    arr[2:(bond_dim-1),1:num_angles,1:num_angles] = reshape(p_list,bond_dim-2,num_angles,1) .* reshape(diagm(ones(Float64,num_angles)),1,num_angles,num_angles)


    return ITensor(arr,righti,upi,downi)
end

"""
    build_mpo_constr_nn_end(q_list, lefti, upi, downi)

Creates the end tensor of the MPO for nearest-neighbor constraints, completing 
the MPO chain by applying any penalties for constraint violations.

# Arguments
- `q_list::Matrix{<:Union{Int,Float64}}`: Matrix specifying penalty values for each 
  ply state pair.
- `lefti::Index`: Bond dimension for the link from the previous tensor.
- `upi::Index`, `downi::Index`: Physical indices representing ply angle states.

# Returns
- `ITensor`: Tensor encoding the end of the MPO constraint.
"""
function build_mpo_constr_nn_end(
    q_list::Matrix{<:Union{Int,Float64}},
    lefti::Index, upi::Index, downi::Index
)::ITensor
    num_angles = size(q_list,2)
    bond_dim = size(q_list,1) + 2

    base_arr = zeros(Float64,bond_dim)
    base_arr[bond_dim] = 1.

    arr = reshape(
        base_arr,bond_dim,1,1
    ) .* reshape(
        diagm(ones(Float64,num_angles)),1,num_angles,num_angles
    )

    arr[2:(bond_dim-1),1:num_angles,1:num_angles] = reshape(
        q_list,bond_dim-2,num_angles,1
    ) .* reshape(diagm(ones(Float64,num_angles)),1,num_angles,num_angles)

    return ITensor(arr,lefti,upi,downi)
end

"""
    build_mpo_constr_nn(sites, p_list, q_list)

Builds the full MPO for enforcing a nearest-neighbor constraint on ply angles 
across the laminate structure. The constraint is defined by `p_list` and `q_list`, 
where `p_list[:, s1] ⋅ q_list[:, s2]` determines if a penalty is applied for 
neighboring ply states `s1` and `s2`.

# Arguments
- `sites::Vector{<:Index}`: Site indices for the MPO corresponding to the ply ngles
    of the laminate.
- `p_list::Matrix{<:Union{Int,Float64}}`: Matrix defining the constraint check vectors 
  for each ply angle.
- `q_list::Matrix{<:Union{Int,Float64}}`: Matrix specifying penalties for each ply angle pair.

# Returns
- `MPO`: The MPO representation of the nearest-neighbor constraint Hamiltonian.
"""
function build_mpo_constr_nn(
    sites::Vector{<:Index}, p_list::Matrix{<:Union{Int,Float64}}, q_list::Matrix{<:Union{Int,Float64}}
)::MPO
    num_sites = length(sites)
    @assert num_sites > 1
    @assert size(p_list) == size(q_list)
    bond_dim = size(p_list,1)+2
    links = [Index(bond_dim, "Link,l=$n") for n ∈ 1:(num_sites-1)]
    Hmpo = MPO(sites)
  
    # first site
    Hmpo[1] = build_mpo_constr_nn_begin(p_list,links[1],sites[1],sites[1]')
    
    # last site
    Hmpo[end] = build_mpo_constr_nn_end(q_list,links[end],sites[end],sites[end]')
  
    if num_sites == 2
      return Hmpo
    end
  
    # second site
    Hmpo[2] = build_mpo_constr_nn_middle(p_list,q_list,links[1],links[2],sites[2],sites[2]')
  
    # fill up other sites with copies
    for n ∈ 3:(num_sites-1)
      Hmpo[n] = replaceinds!(copy(Hmpo[2]),[links[1],links[2],sites[2],sites[2]'],[links[n-1],links[n],sites[n],sites[n]'])
    end
  
    return Hmpo
end

"""
    angles_diff(a1, a2) -> Real

Calculates the absolute difference between two ply angles, normalized to fall 
within the range [0, 90] degrees. This function is useful for determining 
if a disorientation constraint between two angles is violated.

# Arguments
- `a1::Real`, `a2::Real`: Ply angles in degrees.

# Returns
- `Real`: Minimum angle difference within 0 to 90 degrees.
"""
function angles_diff(a1::Real,a2::Real)
    d = abs(a1 - a2) % 180 # 360
    if d > 90 # 180
        return 180 - d # 360
    end
    return d
end

"""
    generate_disorientation_constraint_pq_list(angles, distance, penalty)

Generates `p_list` and `q_list` matrices for the disorientation constraint.
For any two adjacent ply angles `s1` and `s2`, the penalty `penalty` is
applied if `|angles[s1] - angles[s2]| > distance`.

# Arguments
- `angles::Vector{<:Union{Int,Float64}}`: Available ply angles in degrees.
- `distance::Union{Float64, Int}`: Maximum allowable angle difference for disorientation.
- `penalty::Float64`: Penalty value to apply if the constraint is violated.

# Returns
- `Tuple{Matrix, Matrix}`: 
    - `p_list`: Matrix of shape `(1, num_angles)` with a single non-zero entry to 
      indicate the ply angle.
    - `q_list`: Matrix of shape `(1, num_angles)` indicating whether the constraint 
      is violated (penalty applied) based on the neighboring ply angle.
"""
function generate_disorientation_constraint_pq_list(
    angles::Vector{<:Union{Int,Float64}},distance::Union{Float64,Int},penalty::Float64
)
    num_angles = length(angles)
    p_list = diagm(ones(num_angles))
    q_list = ifelse.(angles_diff.(angles, angles') .> distance, penalty, 0.0)

    return p_list,q_list
end