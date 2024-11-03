"""
    build_mpo_constr_knearest_middle(p_lists, lefti, righti, upi, downi)

Constructs the middle tensor for the k-nearest neighbor constraint MPO. This 
constraint applies a penalty based on a generalized form of the nearest-neighbor 
constraint where a sequence of k consecutive plies must satisfy a condition defined by `p_lists`.

Each tensor operation includes the elementwise product of k vectors in `p_lists`, 
where `sum(p1(s1) .* p2(s2) .* ... .* pk(sk))` determines whether the penalty applies 
for the k nearest neighbors.

# Arguments
- `p_lists::Array{<:Union{Int,Float64},3}`: Array of shape `(k, vec_length, num_angles)`, 
  where each `p_lists[i, :, :]` contains the i-th vectors for enforcing the k-nearest 
  constraint among angles.
- `lefti::Index`, `righti::Index`: Bond dimensions for linking MPO tensors.
- `upi::Index`, `downi::Index`: Physical indices representing ply angle states.

# Returns
- `ITensor`: The middle tensor for the MPO constraint.
"""
function build_mpo_constr_knearest_middle(p_lists::Array{<:Union{Int,Float64},3},
    lefti::Index, righti::Index, upi::Index, downi::Index
)::ITensor
    # p_lists = (p1,...pk), shape (k,vec_length,num_angles)
    num_angles = size(p_lists,3)
    k = size(p_lists,1)
    vec_length = size(p_lists,2)
    bond_dim = (k-1)*vec_length + 2

    base_vec = zeros(Float64,bond_dim)
    base_vec[1] = base_vec[bond_dim] = 1.

    base_arr = reshape(
        diagm(base_vec),bond_dim,bond_dim,1
    ) .* reshape(
        ones(Float64,num_angles),1,1,num_angles
    )

    base_arr[1,2:(vec_length+1),:] = p_lists[1,:,:]

    kv = (k-2)*vec_length

        base_arr[2:(bond_dim-vec_length-1),(vec_length+2):(bond_dim-1),:] = reshape(
        permutedims(p_lists[2:(k-1),:,:],(2,1,3)),kv,1,num_angles
    ) .* reshape(diagm(ones(Float64,kv)),kv,kv,1)
    
    base_arr[(bond_dim-vec_length):(bond_dim-1),bond_dim,:] = p_lists[k,:,:]

    arr = reshape(
        base_arr,bond_dim,bond_dim,num_angles,1
    ) .* reshape(
        diagm(ones(Float64,num_angles)),1,1,num_angles,num_angles
    )

    return ITensor(arr,lefti,righti,upi,downi)
end

"""
    build_mpo_constr_knearest_begin(p_lists, righti, upi, downi)

Constructs the beginning tensor for the k-nearest neighbor constraint MPO. 
The tensor starts the MPO chain by evaluating the k-nearest constraint on 
the first site with `p_lists[1, :, :]`.

# Arguments
- `p_lists::Array{<:Union{Int,Float64},3}`: Array defining the k-nearest constraint vectors 
  for each ply angle.
- `righti::Index`: Bond dimension for linking to the next tensor.
- `upi::Index`, `downi::Index`: Physical indices representing ply angle states.

# Returns
- `ITensor`: The beginning tensor for the MPO constraint.
"""
function build_mpo_constr_knearest_begin(
    p_lists::Array{<:Union{Int,Float64},3},
    righti::Index, upi::Index, downi::Index
)::ITensor
    num_angles = size(p_lists,3)
    k = size(p_lists,1)
    vec_length = size(p_lists,2)
    bond_dim = (k-1)*vec_length + 2
    base_vec = zeros(Float64,bond_dim)
    base_vec[1] = 1.

    base_arr = reshape(
        base_vec,bond_dim,1
    ) .* reshape(
        ones(Float64,num_angles),1,num_angles
    )

    base_arr[2:(vec_length+1),:] = p_lists[1,:,:]

    arr = reshape(
        base_arr,bond_dim,num_angles,1
    ) .* reshape(diagm(ones(Float64,num_angles)),1,num_angles,num_angles)

    return ITensor(arr,righti,upi,downi)
end

"""
    build_mpo_constr_knearest_end(p_lists, lefti, upi, downi)

Constructs the end tensor for the k-nearest neighbor constraint MPO. 
This tensor concludes the MPO chain by applying any penalties for 
constraint violations.

# Arguments
- `p_lists::Array{<:Union{Int,Float64},3}`: Array defining the constraint vectors for 
  each ply angle.
- `lefti::Index`: Bond dimension for linking from the previous tensor.
- `upi::Index`, `downi::Index`: Physical indices representing ply angle states.

# Returns
- `ITensor`: The end tensor for the MPO constraint.
"""
function build_mpo_constr_knearest_end(
    p_lists::Array{<:Union{Int,Float64},3},
    righti::Index, upi::Index, downi::Index
)::ITensor
    num_angles = size(p_lists,3)
    k = size(p_lists,1)
    vec_length = size(p_lists,2)
    bond_dim = (k-1)*vec_length + 2
    base_vec = zeros(Float64,bond_dim)
    base_vec[bond_dim] = 1.

    base_arr = reshape(
        base_vec,bond_dim,1
    ) .* reshape(
        ones(Float64,num_angles),1,num_angles
    )

    base_arr[(bond_dim-vec_length):(bond_dim-1),:] = p_lists[k,:,:]

    arr = reshape(
        base_arr,bond_dim,num_angles,1
    ) .* reshape(diagm(ones(Float64,num_angles)),1,num_angles,num_angles)

    return ITensor(arr,righti,upi,downi)
end

"""
    build_mpo_constr_knearest(sites, p_lists)

Builds the complete MPO for enforcing the k-nearest neighbor constraint across 
all sites. This MPO can apply penalties if the k nearest neighbors do not satisfy 
the constraint defined by `p_lists`.

Each tensor operation includes the elementwise product of k vectors in `p_lists`, 
where `sum(p1(s1) .* p2(s2) .* ... .* pk(sk))` determines whether the penalty applies 
for the k nearest neighbors.

# Arguments
- `sites::Vector{<:Index}`: Site indices of the MPO, corresponding to angle for each ply 
    in the laminate.
- `p_lists::Array{<:Union{Int,Float64},3}`: Array of shape `(k, vec_length, num_angles)`, 
  where each `p_lists[i, :, :]` contains the i-th vectors for enforcing the k-nearest 
  constraint among angles.

# Returns
- `MPO`: The MPO representation of the k-nearest neighbor constraint Hamiltonian.
"""
function build_mpo_constr_knearest(sites::Vector{<:Index}, p_lists::Array{<:Union{Int,Float64},3})
    num_sites = length(sites)
    @assert num_sites > 1
    k = size(p_lists,1)
    vec_length = size(p_lists,2)
    bond_dim = (k-1)*vec_length + 2
    links = [Index(bond_dim, "Link,l=$n") for n ∈ 1:(num_sites-1)]
    Hmpo = MPO(sites)
  
    # first site
    Hmpo[1] = build_mpo_constr_knearest_begin(p_lists,links[1],sites[1],sites[1]')
    
    # last site
    Hmpo[end] = build_mpo_constr_knearest_end(p_lists,links[end],sites[end],sites[end]')
  
    if num_sites == 2
      return Hmpo
    end
  
    # second site
    Hmpo[2] = build_mpo_constr_knearest_middle(p_lists,links[1],links[2],sites[2],sites[2]')
  
    # fill up other sites with copies
    for n ∈ 3:(num_sites-1)
      Hmpo[n] = replaceinds!(copy(Hmpo[2]),[links[1],links[2],sites[2],sites[2]'],[links[n-1],links[n],sites[n],sites[n]'])
    end
  
    return Hmpo
  end

"""
    generate_plists_contiguity(num_angles, num_contiguity, γ)

Generates `p_lists` for the contiguity constraint as a special case of the 
k-nearest constraint. This constraint applies a penalty when `num_contiguity + 1` 
consecutive plies have the same angle. The `p_lists` elements indicate the same 
angle across k nearest neighbors, with the final vector in `p_lists` weighted by 
`γ` to specify the penalty if the contiguity constraint is violated.

# Arguments
- `num_angles::Int`: Total number of possible ply angles.
- `num_contiguity::Int`: Number of consecutive plies for enforcing contiguity.
- `γ::Float64`: Penalty applied when the contiguity constraint is violated.

# Returns
- `Array{Float64, 3}`: Array of shape `(k, vec_length, num_angles)` where each 
  row vector in `p_lists` enforces ply angle continuity and the final vector 
  represents the penalty if k consecutive plies share the same angle.
"""
function generate_plists_contiguity(num_angles::Int,num_contiguity::Int,γ::Float64)
    # size(arr) = (k,vec_length,num_angles), 
    # with here vec_length = num_angles, k = num_contiguity + 1
    k = num_contiguity + 1
    base_vec = ones(Float64, k)
    base_vec[k] = γ
    arr = reshape(base_vec,k,1,1) .* reshape(
        diagm(ones(Float64,num_angles)),1,num_angles,num_angles
    )

    return arr
end