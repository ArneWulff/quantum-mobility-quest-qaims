"""
    generate_loss_local_eigenvalues(
        laminate::Laminate,
        target::Matrix{Float64},
        distribute_target::String="proportion"
    )

Generates local eigenvalues that represent the difference between the 
current lamination parameters of a `Laminate` and a target parameter 
distribution. These local eigenvalues will be used as part of the Hamiltonian 
construction for the loss function, encapsulating the distance to target 
lamination parameters.

The function can distribute the target parameter values across the sum over 
plies in two ways, specified by `distribute_target`:
  - `"even"`: Divides the target values evenly across plies, making the 
    target term in each ply `target / num_plies`.
  - `"proportion"`: Distributes the target according to each weight in 
    `laminate.weights`, so that the target scales proportionally with ply 
    weight values.

# Arguments
- `laminate::Laminate`: The `Laminate` instance with ply weights and functions.
- `target::Matrix{Float64}`: A matrix of target values with dimensions 
  `(num_weights, num_funcs)`, matching the output of `parameters(laminate, stack)`.
- `distribute_target::String`: Specifies the mode for target distribution; 
   `"even"` (default) or `"proportion"`.

# Returns
- `Array{Float64, 4}`: A 4D array of eigenvalues with dimensions 
  `(num_weights, num_plies, num_funcs, num_angles)` for constructing the MPO.
"""

function generate_loss_local_eigenvalues(
    laminate::Laminate,
    target::Matrix{Float64},
    distribute_target::String="proportion"
)::Array{Float64, 4}
    @assert distribute_target in ["even","proportion"]

    f = reshape(transpose(laminate.funcs),
                1,1,num_funcs(laminate),num_angles(laminate))
    w = reshape(laminate.weights,
                num_weights(laminate),num_plies(laminate),1,1)
    t = reshape(target,
                num_weights(laminate),1,num_funcs(laminate),1)
    return distribute_target == "even" ? w .* f .- t : w .* (f .- t)
end


"""
    build_mpo_loss_middle(
        X::Int, l::Int, n::Int, ev_array::Array{Float64, 4},
        lefti::Index, righti::Index, upi::Index, downi::Index
    )

Builds the MPO tensor at a middle site for the loss function Hamiltonian.
This tensor operates on the `n`-th ply, indexed by `upi` and `downi`, 
which represent possible ply states (e.g., 0°, 45°, 90°, -45°).

The loss function Hamiltonian uses the local eigenvalues from `ev_array` to 
define the target deviation at each ply angle. This middle-site tensor is 
connected to adjacent sites by `lefti` and `righti`.

# Arguments
- `X::Int`: Index of the lamination weight type (e.g., 1 for `A`, 2 for `B`, 3 for `D`).
- `l::Int`: Index of the angle function type (e.g., 1 for `cos(2θ)`, 2 for `sin(2θ)`).
- `n::Int`: The site index in the MPO (specifically, a middle ply).
- `ev_array::Array{Float64, 4}`: 4D array of eigenvalues with dimensions 
   `(num_weights, num_plies, num_funcs, num_angles)`, generated by 
   `generate_loss_local_eigenvalues`.
- `lefti::Index`, `righti::Index`: Link indices to connect to adjacent tensors.
- `upi::Index`, `downi::Index`: Ply state indices representing angle states.

# Returns
- `ITensor`: The MPO tensor for the middle site.
"""
function build_mpo_loss_middle(X::Int,l::Int,n::Int,ev_array::Array{Float64,4},
    lefti::Index, righti::Index, upi::Index, downi::Index
)::ITensor
    num_angles = size(ev_array, 4)
    arr = reshape(
        diagm(ones(Float64,3)),3,3,1,1
    ) .* reshape(
        diagm(ones(Float64,num_angles)),1,1,num_angles,num_angles
    )

    arr[CartesianIndex.(
        fill(1,num_angles),fill(2,num_angles),1:num_angles,1:num_angles
    )] = arr[
        CartesianIndex.(
        fill(2,num_angles),fill(3,num_angles),1:num_angles,1:num_angles
    )
    ] = sqrt(2) .* ev_array[X,n,l,1:num_angles]

    arr[CartesianIndex.(
        fill(1,num_angles),fill(3,num_angles),1:num_angles,1:num_angles
    )] = ev_array[X,n,l,1:num_angles] .^ 2

    return ITensor(arr,lefti,righti,upi,downi)
end

"""
    build_mpo_loss_begin(
        X::Int, l::Int, ev_array::Array{Float64, 4},
        righti::Index, upi::Index, downi::Index
    )

Builds the MPO tensor for the beginning (first) site of the loss function 
Hamiltonian, with initial conditions that match the first ply state.

# Arguments
- `X::Int`: Index of the lamination weight type (e.g., 1 for `A`, 2 for `B`, 3 for `D`).
- `l::Int`: Index of the angle function type (e.g., 1 for `cos(2θ)`, 2 for `sin(2θ)`).
- `ev_array::Array{Float64, 4}`: 4D array of eigenvalues with dimensions 
   `(num_weights, num_plies, num_funcs, num_angles)`.
- `righti::Index`: Link index to the next site.
- `upi::Index`, `downi::Index`: Ply state indices representing angle states.

# Returns
- `ITensor`: The MPO tensor for the first site.
"""
function build_mpo_loss_begin(X::Int, l::Int, ev_array::Array{Float64,4},
    righti::Index, upi::Index, downi::Index
)::ITensor
    num_angles = size(ev_array, 4)
    arr = reshape(
        [1.,0.,0.],3,1,1
    ) .* reshape(
        diagm(ones(Float64,num_angles)),1,num_angles,num_angles
    )

    arr[CartesianIndex.(
        fill(2,num_angles),1:num_angles,1:num_angles
    )] = sqrt(2) .* ev_array[X,1,l,1:num_angles]

    arr[CartesianIndex.(
        fill(3,num_angles),1:num_angles,1:num_angles
    )] = ev_array[X,1,l,1:num_angles] .^ 2

    return ITensor(arr,righti,upi,downi)
end

"""
    build_mpo_loss_end(
        X::Int, l::Int, ev_array::Array{Float64, 4},
        lefti::Index, upi::Index, downi::Index
    )

Builds the MPO tensor for the last site of the loss function Hamiltonian, 
representing the last ply state. This tensor completes the MPO chain.

# Arguments
- `X::Int`: Index of the lamination weight type (e.g., 1 for `A`, 2 for `B`, 3 for `D`).
- `l::Int`: Index of the angle function type (e.g., 1 for `cos(2θ)`, 2 for `sin(2θ)`).
- `ev_array::Array{Float64, 4}`: 4D array of eigenvalues with dimensions 
   `(num_weights, num_plies, num_funcs, num_angles)`.
- `lefti::Index`: Link index from the previous site.
- `upi::Index`, `downi::Index`: Ply state indices representing angle states.

# Returns
- `ITensor`: The MPO tensor for the last site.
"""
function build_mpo_loss_end(X::Int, l::Int, ev_array::Array{Float64,4},
    lefti::Index, upi::Index, downi::Index
)::ITensor
    num_angles = size(ev_array, 4)
    num_plies = size(ev_array, 2)
    arr = reshape(
        [0.,0.,1.],3,1,1
    ) .* reshape(
        diagm(ones(Float64,num_angles)),1,num_angles,num_angles
    )

    arr[CartesianIndex.(
        fill(2,num_angles),1:num_angles,1:num_angles
    )] = sqrt(2) .* ev_array[X,num_plies,l,1:num_angles]

    arr[CartesianIndex.(
        fill(1,num_angles),1:num_angles,1:num_angles
    )] = ev_array[X,num_plies,l,1:num_angles] .^ 2

    return ITensor(arr,lefti,upi,downi)
end

"""
    build_mpo_loss(
        X::Int, l::Int,
        sites::Vector{<:Index}, ev_array::Array{Float64, 4}
    )

Constructs the entire MPO for the loss function Hamiltonian, given the 
lamination weight and angle function types specified by `X` and `l`. 
The MPO tensor chain applies to the entire set of plies, generating a 
Hamiltonian that represents the squared distance between each ply's 
parameters and the target lamination parameters.

# Arguments
- `X::Int`: Index of the lamination weight type (e.g., 1 for `A`, 2 for `B`, 3 for `D`).
- `l::Int`: Index of the angle function type (e.g., 1 for `cos(2θ)`, 2 for `sin(2θ)`).
- `sites::Vector{<:Index}`: Indices for each ply in the tensor chain.
- `ev_array::Array{Float64, 4}`: 4D array of eigenvalues with dimensions 
   `(num_weights, num_plies, num_funcs, num_angles)`.

# Returns
- `MPO`: The completed MPO representing the loss function Hamiltonian.
"""
function build_mpo_loss(X::Int, l::Int,
    sites::Vector{<:Index}, ev_array::Array{Float64,4}
)::MPO
    num_plies = size(ev_array, 2)
    # num_states = size(ev_array, 4)
    links = [Index(3,"Link,l=$n") for n ∈ 1:(num_plies-1)]
    Hmpo = MPO(sites)
    for n in 1:num_plies
        if n > 1
            lefti = links[n-1]
        end
        if n < num_plies
            righti = links[n]
        end
        site = sites[n]
        upi = site
        downi = site'
        if n == 1
            T = build_mpo_loss_begin(X, l, ev_array, righti, upi, downi)
        elseif n == num_plies
            T = build_mpo_loss_end(X, l, ev_array, lefti, upi, downi)
        else
            T = build_mpo_loss_middle(X, l, n, ev_array, lefti, righti, upi, downi)
        end
        Hmpo[n] = T
    end
    return Hmpo
end
