"""
    build_mpo(sites::Vector{<:Index}, laminate::Laminate, target::Matrix{Float64}, 
              constraints::ConstraintSettings, 
              exclude_pars::Union{Vector{Tuple{Int,Int}}, Tuple{Int,Int}, Nothing}=nothing)

Constructs a list of Matrix Product Operators (MPOs) representing the target lamination parameter 
objectives and any active constraints. The MPO list includes:
  - A loss function MPO, representing the lamination parameter difference from `target`.
  - Constraint MPOs, if constraints are specified in `ConstraintSettings`, including 
    nearest-neighbor, k-nearest, balanced, and minimum constraints.

Arguments:
  - `sites::Vector{<:Index}`: Index vector defining the MPS sites for the MPOs.
  - `laminate::Laminate`: The laminate structure with properties such as weights and functions.
  - `target::Matrix{Float64}`: Target lamination parameters.
  - `constraints::ConstraintSettings`: Settings for constraints that may apply penalties to 
    the loss function.
  - `exclude_pars::Union{Vector{Tuple{Int,Int}}, Tuple{Int,Int}, Nothing}`: Parameter indices 
    to exclude from the loss function, which is useful for tuning specific parameters.

Returns:
  - `Vector{MPO}`: A vector of MPOs that represent the overall Hamiltonian, including the 
    loss function and any specified constraints.
"""
function build_mpo(
    sites::Vector{<:Index}, laminate::Laminate, target::Matrix{Float64}, 
    constraints::ConstraintSettings, 
    exclude_pars::Union{Vector{Tuple{Int,Int}}, Tuple{Int,Int}, Nothing} = nothing
)::Vector{MPO}
    if exclude_pars === nothing
        exclude_pars = Tuple{Int,Int}[]
    elseif typeof(exclude_pars) == Tuple{Int,Int}
        exclude_pars = [exclude_pars]
    end
    ev_array = generate_loss_local_eigenvalues(laminate,target)
    
    mpo_loss = vec([
        build_mpo_loss(X,l,sites,ev_array) 
        for X in 1:num_weights(laminate), l in 1:num_funcs(laminate)
        if (X,l) ∉ exclude_pars
    ])
    
    mpo_nn = constraints.nn ? [
            build_mpo_constr_nn(sites, constraints.nn_p_list, constraints.nn_q_list)
    ] : MPO[]
    
    mpo_knearest = constraints.knearest ? [
        build_mpo_constr_knearest(sites,constraints.knearest_plists)
    ] : MPO[]

    mpo_balanced = constraints.balanced ? [
        build_mpo_constr_balanced(
            sites, num_angles(laminate),s,t,constraints.balanced_penalty
        ) for (s,t) in constraints.balanced_angles
    ] : MPO[]

    mpo_minimum = constraints.minimum ? [
        build_mpo_constr_minimum(
            sites,num_angles(laminate),t,t_min,constraints.minimum_penalty
        ) for (t,t_min) in constraints.minimum_t
    ] : MPO[]

    return [mpo_loss;mpo_nn;mpo_knearest;mpo_balanced;mpo_minimum]
end


"""
    DMRGResult

Struct to store the results from a DMRG computation.

Fields:
  - `psi::MPS`: The final optimized Matrix Product State (MPS) after DMRG convergence.
  - `energies::Vector{Float64}`: A record of energy values at each sweep.
  - `times::Vector{Float64}`: Cumulative times at the end of each sweep.
  - `sweep_durations::Vector{Float64}`: Time taken per sweep.
  - `max_linkdims::Vector{Int}`: Maximum bond dimension reached during each sweep.

"""
struct DMRGResult
    psi::MPS
    energies::Vector{Float64}
    times::Vector{Float64}
    sweep_durations::Vector{Float64}
    max_linkdims::Vector{Int}
end


"""
    gen_bond_dims(num_sweeps::Int, max_bond_dim::Int) -> Vector{Int}

Generates a list of maximum bond dimensions for a series of DMRG sweeps, starting with a constant 
bond dimension and progressively reducing it in the final sweeps to produce a basis state.

The bond dimension remains constant at `max_bond_dim` throughout the majority of the sweeps. 
In the final stages, the bond dimension is halved at each sweep, culminating in a bond dimension 
of 1 on the second-to-last sweep. This ensures a product state by the final sweep, effectively 
collapsing any remaining superposition into a single basis (stacking) state.

Arguments:
  - `num_sweeps::Int`: Total number of DMRG sweeps.
  - `max_bond_dim::Int`: The maximum bond dimension to maintain until the final sweeps.

Returns:
  - `Vector{Int}`: A list of maximum bond dimensions for each sweep.
"""
function gen_bond_dims(num_sweeps::Int,max_bond_dim::Int)
    decrease_bd = [max_bond_dim]
    while decrease_bd[end] > 1
        push!(decrease_bd,decrease_bd[end]÷2)
    end
    push!(decrease_bd,1)
    return append!(
        fill(max_bond_dim,num_sweeps-length(decrease_bd)),decrease_bd
    )
end

"""
    gen_cutoffs(num_sweeps::Int, co::Float64=1e-14) -> Vector{Float64}

Generates a list of cutoffs for singular values during each DMRG sweep, with a high cutoff in 
the last sweep to enforce a basis state.

The cutoff remains fixed at `co` for the majority of the sweeps, allowing for standard convergence. 
On the final sweep, the cutoff is raised to 0.5, which forces the DMRG to eliminate any remaining 
superposition in favor of a single stacking sequence basis state.

Arguments:
  - `num_sweeps::Int`: Total number of sweeps.
  - `co::Float64=1e-14`: Default cutoff value for all but the final sweep.

Returns:
  - `Vector{Float64}`: A list of cutoffs for each sweep.
"""
gen_cutoffs(num_sweeps::Int,co::Float64=1e-14) = append!(zeros(Float64,num_sweeps-1).+co,[0.5])


"""
    energy_Hsum_psi(Hlist::Vector{<:MPO}, psi::MPS)

Calculates the expectation value of a Hamiltonian represented by a sum of MPOs, `Hlist`, 
in the MPS `psi`.

Arguments:
  - `Hlist`: Can be either a `Vector{MPO}` or a `ProjMPOSum`, representing a sum
    of Hamiltonians.
  - `psi::MPS`: The Matrix Product State (MPS) used to calculate the expectation value.

Returns:
  - `Float64`: The calculated energy.
"""
energy_Hsum_psi(Hlist::Vector{<:MPO},psi::MPS) = sum([inner(psi',H,psi) for H ∈ Hlist])


function energy_Hsum_psi(PH::ProjMPOSum,psi::MPS)::Float64
    terms = :terms ∈ fieldnames(ProjMPOSum) ? PH.terms : PH.pm
    return sum([inner(psi',p.H,psi) for p ∈ terms])
end

"""
    dmrg_custom(PH::ProjMPOSum, psi0::MPS, sweeps::Sweeps, sweep_sequence::String; kwargs...) -> DMRGResult

Performs a DMRG optimization using a modified sweep pattern. This function extends the standard
ITensors `dmrg` function, adding flexibility to control the sweep direction, which may be defined 
by `sweep_sequence`. 

Arguments:
  - `PH::ProjMPOSum`: The projected MPO sum representing the Hamiltonian.
  - `psi0::MPS`: The initial MPS.
  - `sweeps::Sweeps`: The sweep settings, defining parameters like maximum bond dimension, cutoff, 
    and noise level.
  - `sweep_sequence::String`: A sequence defining the sweep direction for each sweep (e.g., 
    "LLRR" for two left-to-right and two right-to-left sweeps).

Keyword Arguments:
  - `eigsolve_tol::Number=1e-14`: Tolerance for eigenvalue solver.
  - `eigsolve_krylovdim::Int=3`: Maximum dimension of the Krylov space.
  - `eigsolve_maxiter::Int=1`: Maximum number of iterations for the eigenvalue solver.
  - `outputlevel::Int=1`: Verbosity of output. Higher values provide more detail.

Returns:
  - `DMRGResult`: A result struct containing the optimized MPS (`psi`), energy history, 
    sweep times, and maximum bond dimensions across sweeps.

Notes:
  This implementation adapts the ITensors `dmrg` method to allow a customizable sweep 
  direction sequence, enabling optimization patterns beyond standard alternating sweeps.
"""
function dmrg_custom(
    PH::ProjMPOSum,psi0::MPS,
    sweeps::Sweeps,sweep_sequence::String;
    kwargs...
)   
    @assert length(PH.terms[1]) == length(psi0)
    num_plies = length(psi0)
    svd_alg = "divide_and_conquer"
    obs = NoObserver()
    write_when_maxdim_exceeds = nothing

    outputlevel::Int = get(kwargs, :outputlevel, 1)

    # eigsolve kwargs
    eigsolve_tol::Number = get(kwargs, :eigsolve_tol, 1e-14)
    eigsolve_krylovdim::Int = get(kwargs, :eigsolve_krylovdim, 3)
    eigsolve_maxiter::Int = get(kwargs, :eigsolve_maxiter, 1)
    eigsolve_verbosity::Int = get(kwargs, :eigsolve_verbosity, 0)

    ishermitian::Bool = get(kwargs, :ishermitian, true)

    eigsolve_which_eigenvalue::Symbol = :SR

    psi = copy(psi0)
    N = length(psi)

    PH = copy(PH)

    first_site = sweep_sequence[1] == 'L' ? length(psi) - 1 : 1 # orthogonalize to left of the two sites (-1)?
    if !isortho(psi) || ITensors.orthocenter(psi) != first_site
        orthogonalize!(psi, first_site)
    end
    @assert isortho(psi) && ITensors.orthocenter(psi) == first_site

    position!(PH, psi, first_site)
    energy = 0.0

    len_sweep_sequence = length(sweep_sequence)

    last_sweep_direction = 'X' # something different than 'L' and 'R', since already orthogonalized

    num_sweeps = length(sweeps)
    energies_list = Vector{Float64}(undef,num_sweeps)
    times_list = Vector{Float64}(undef,num_sweeps)
    elapsed_times_sweeps = Vector{Float64}(undef,num_sweeps)
    maxlinkdim_list = Vector{Int}(undef,num_sweeps)

    t0 = time()
    for sw in 1:nsweep(sweeps)
        sweep_direction = sweep_sequence[(sw-1)%len_sweep_sequence+1]
        
        sw_time = @elapsed begin
            maxtruncerr = 0.0

            first_site = sweep_direction == 'L' ? length(psi) - 1 : 1 # orthogonalize to left of the two sites
            if sweep_direction == last_sweep_direction
                orthogonalize!(psi, first_site)
            end
            
            iterator = sweep_direction == 'L' ? sweepnext_to_left(num_plies) : sweepnext_to_right(num_plies)
            for (b, ha) in iterator

                position!(PH, psi, b)

                phi = psi[b] * psi[b+1]

                vals, vecs = eigsolve(
                    PH,
                    phi,
                    1,
                    eigsolve_which_eigenvalue;
                    ishermitian=ishermitian,
                    tol=eigsolve_tol,
                    krylovdim=eigsolve_krylovdim,
                    maxiter=eigsolve_maxiter
                )

                energy = vals[1]
                phi::ITensor = vecs[1]

                ortho = ha == 1 ? "left" : "right"

                drho = nothing
                if noise(sweeps, sw) > 0.0
                    drho = noise(sweeps, sw) * noiseterm(PH, phi, ortho)
                end


                spec = replacebond!(
                    psi,
                    b,
                    phi;
                    maxdim=maxdim(sweeps, sw),
                    mindim=mindim(sweeps, sw),
                    cutoff=cutoff(sweeps, sw),
                    eigen_perturbation=drho,
                    ortho=ortho,
                    normalize=true,
                    svd_alg=svd_alg
                )
                maxtruncerr = max(maxtruncerr, spec.truncerr)

                if outputlevel >= 2
                    @printf("Sweep %d, half %d, bond (%d,%d) energy=%s\n", sw, ha, b, b + 1, energy)
                    @printf(
                        "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
                        cutoff(sweeps, sw),
                        maxdim(sweeps, sw),
                        mindim(sweeps, sw)
                    )
                    @printf(
                        "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
                    )
                    flush(stdout)
                end

                if sweep_direction == 'R'
                    sweep_is_done = (b == N-1 && ha == 1)
                else
                    sweep_is_done = (b == 1 && ha == 2)
                end
                measure!(
                    obs;
                    energy=energy,
                    psi=psi,
                    bond=b,
                    sweep=sw,
                    half_sweep=ha,
                    spec=spec,
                    outputlevel=outputlevel,
                    sweep_is_done=sweep_is_done
                )
            end
        end

        t1 = time() - t0

        if outputlevel >= 1
            @printf(
                "After sweep %d %s energy=%s  maxlinkdim=%d maxerr=%.2E time=%.3f\n",
                sw,
                sweep_direction,
                energy,
                maxlinkdim(psi),
                maxtruncerr,
                sw_time
            )
            flush(stdout)
        end

        energies_list[sw] = energy
        times_list[sw] = t1
        elapsed_times_sweeps[sw] = sw_time
        maxlinkdim_list[sw] = maxlinkdim(psi)
        
        last_sweep_direction = sweep_direction
        
        isdone = checkdone!(obs; energy=energy, psi=psi, sweep=sw, outputlevel=outputlevel)
        isdone && break
    end

    return DMRGResult(psi,energies_list,times_list,elapsed_times_sweeps,maxlinkdim_list)

end