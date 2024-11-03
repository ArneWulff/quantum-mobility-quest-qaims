"""
    result_psi_to_stack(psi::MPS)

Convert the MPS `psi` after the final sweep into a stacking sequence, assuming it represents 
a collapsed (basis) state. Each tensor is projected into its most probable state by evaluating 
and identifying the maximum amplitude at each ply.

Returns:
  - `Vector{Int}`: The stacking sequence as a vector of integers, where each entry corresponds 
    to a ply angle as determined by the MPS.
"""
function result_psi_to_stack(psi::MPS)::Vector{Int}
    stack = Vector{Int}(undef,length(psi))
    sites = siteinds(psi)
    for (n,t) ∈ enumerate(psi)
        if n == 1
            bi = commonind(t,psi[n+1])
            stack[n] = argmax([t[bi=>1,sites[n]=>s]^2 for s ∈ 1:dim(sites[n])])
            continue
        elseif n == length(psi)
            bi = commonind(t,psi[n-1])
            stack[n] = argmax([t[bi=>1,sites[n]=>s]^2 for s ∈ 1:dim(sites[n])])
            continue
        end
        bi1 = commonind(t,psi[n-1])
        bi2 = commonind(t,psi[n+1])
        stack[n] = argmax([t[bi1=>1,bi2=>1,sites[n]=>s]^2 for s ∈ 1:dim(sites[n])])
    end
    return stack
end

"""
    save_value_to_hdf5!(parent::Union{HDF5.File, HDF5.Group}, key::String, val::Any)

Save a single value `val` to an HDF5 file or group under `key`. Handles individual values 
(Number, Bool, or String) and arrays of these types. Other types are converted to a string 
before saving.

Arguments:
  - `parent`: HDF5 file or group in which the value will be saved.
  - `key`: Key for the value in the HDF5 structure.
  - `val`: Value to store; if it's not a primitive type or array, it is stored as a string.
"""
function save_value_to_hdf5!(parent::Union{HDF5.File,HDF5.Group},key::String,val::Any)
    if val isa Number || val isa Bool || val isa String
        attributes(parent)[key] = val
    elseif val isa AbstractArray && all(x -> x isa Number || x isa Bool || x isa String, val)
        parent[key] = val
    else
        attributes(parent)[key] = string(val)
    end
end

"""
    save_dict_to_hdf5!(parent::Union{HDF5.File, HDF5.Group}, dict::Dict{String, Any})

Save all key-value pairs from a dictionary `dict` into an HDF5 group or file. Each key becomes 
an attribute or dataset in `parent`, depending on the type of the value.

Arguments:
  - `parent`: HDF5 file or group where the dictionary contents will be saved.
  - `dict`: Dictionary containing key-value pairs to save. Values are processed with 
    `save_value_to_hdf5!`.
"""
function save_dict_to_hdf5!(parent::Union{HDF5.File,HDF5.Group},dict::Dict{String,Any})
    for (key,val) in dict
        save_value_to_hdf5!(parent,key,val)
    end
end

"""
    save_struct_to_hdf5!(parent::Union{HDF5.File, HDF5.Group}, obj::Any)

Convert all fields of a struct `obj` to a dictionary and save them into an HDF5 file or group.
Each struct field is saved as a key-value pair in `parent`, with values processed by 
`save_dict_to_hdf5!`.

Arguments:
  - `parent`: HDF5 file or group where the struct fields will be saved.
  - `obj`: The struct instance whose fields are saved to `parent`.
"""
function save_struct_to_hdf5!(parent::Union{HDF5.File,HDF5.Group},obj::Any)
    save_dict_to_hdf5!(parent, Dict(
        string(f) => getfield(obj, f) for f in fieldnames(typeof(obj))
    ))
end

"""
    create_file_for_dmrg_experiment(filelocation::String, filename::String, sample_idx::Int,
                                    laminate::Laminate, target_parameters::Matrix{Float64},
                                    constraints::ConstraintSettings, sweeps::Sweeps, 
                                    sweep_sequence::String, psi0_energies::Vector{Float64}, 
                                    psi0_list::Union{Vector{MPS}, Nothing} = nothing; 
                                    save_properties::Union{Dict{String, Any}, Nothing} = nothing) 
                                    -> String

Create and initialize an HDF5 file for storing data from a DMRG experiment.

Arguments:
  - `filelocation`: Directory where the file will be created.
  - `filename`: Base name for the HDF5 file.
  - `sample_idx`: Sample identifier.
  - `laminate`: The `Laminate` instance containing laminate properties.
  - `target_parameters`: Matrix of target lamination parameters.
  - `constraints`: `ConstraintSettings` struct defining manufacturing constraints.
  - `sweeps`: Sweep settings for the DMRG process.
  - `sweep_sequence`: String specifying the order of sweeps in DMRG.
  - `psi0_energies`: Vector containing initial MPS energies.
  - `psi0_list`: Optional, initial MPS configurations to store in the file.
  - `save_properties`: Optional dictionary for additional metadata
    (e.g., angles, target stack).

The file is initialized with datasets and groups for storing DMRG results, experiment parameters, 
constraints, and initial configurations.

Returns:
  - `String`: The full file path of the created HDF5 file.
"""
function create_file_for_dmrg_experiment(
    filelocation::String, filename::String, sample_idx::Int,
    laminate::Laminate,target_parameters::Matrix{Float64},
    constraints::ConstraintSettings,
    sweeps::Sweeps,sweep_sequence::String,
    psi0_energies::Vector{Float64},
    psi0_list::Union{Vector{MPS},Nothing} = nothing;
    save_properties::Union{Dict{String,Any},Nothing} = nothing
)::String
    # save_properties: angles, target_stack, constraint details...

    # add backslash at the end of directory, if necessary
    if filelocation[end] ≠ '\\'
        filelocation *= "\\"
    end

    # create complete filepath
    filepath = filelocation*filename*"_sample_"*lpad(sample_idx,4,"0")*".hdf5"

    num_tries = length(psi0_energies)
    num_sweeps = length(sweeps)

    # create HDF5 file
    fid = h5open(filepath,"cw")

    # store properties
    props = create_group(fid,"properties")
    attributes(props)["sample_idx"] = sample_idx
    attributes(props)["num_plies"] = num_plies(laminate)
    attributes(props)["num_angles"] = num_angles(laminate)
    attributes(props)["num_funcs"] = num_funcs(laminate)
    attributes(props)["num_weights"] = num_weights(laminate)
    attributes(props)["sweep_sequence"] = sweep_sequence
    attributes(props)["num_sweeps"] = num_sweeps
    attributes(props)["num_tries"] = num_tries
    # props["angles"] = angles
    props["target_parameters"] = target_parameters
    # props["target_stack"] = target_stack

    group_sweeps = create_group(props,"sweeps")
    group_sweeps["maxdim"] = sweeps.maxdim
    group_sweeps["cutoff"] = sweeps.cutoff
    attributes(props)["maxdim"] = maximum(sweeps.maxdim)

    props_constraints = create_group(props,"constraints")
    save_struct_to_hdf5!(props_constraints,constraints)
    if save_properties !== nothing
        save_dict_to_hdf5!(props,save_properties)
    end
    
    # store psi0
    props["psi0_energies"] = psi0_energies
    if psi0_list !== nothing
        group_psi0 = create_group(fid,"psi0")
        for (n,(psi0,psi0e)) ∈ enumerate(zip(psi0_list,psi0_energies))
            HDF5.write(group_psi0,"try_$n",psi0)
        end
    end

    # create datasets for data and results
    data = create_group(fid,"data")
    
    create_dataset(data,"energies",Float64,(num_tries,num_sweeps))
    create_dataset(data,"time_stamps",Float64,(num_tries,num_sweeps))
    create_dataset(data,"elapsed_time_sweeps",Float64,(num_tries,num_sweeps))
    create_dataset(data,"maxlinkdim",Int,(num_tries,num_sweeps))

    res_group = create_group(fid,"results")
    create_dataset(res_group,"stack",Int,(num_tries,num_plies(laminate)))
    create_dataset(res_group,"lamination_parameters",Float64,(num_tries,num_weights(laminate)*num_funcs(laminate)))
    create_dataset(res_group,"loss",Float64,(num_tries,))
    create_dataset(res_group,"rmse",Float64,(num_tries,))
    create_dataset(res_group,"constraint_violations",Int,(num_tries,4))

    close(fid)

    return filepath
end

"""
    dmrg_experiment(filelocation::String, filename::String, sample_idx::Int, 
                    laminate::Laminate, target_parameters::Matrix{Float64},
                    constraints::ConstraintSettings, sites::Vector{<:Index},
                    PH::ProjMPOSum, psi0_list::Vector{MPS}, num_sweeps::Int, 
                    sweep_sequence::String, max_bond_dim::Int; 
                    save_psi0::Bool=false, save_properties::Union{Dict{String,Any},Nothing}=nothing,
                    target_stack::Union{Vector{Int}, Nothing}=nothing, kwargs...)

Run the DMRG optimization experiment to retrieve stacking sequences that align with target lamination parameters.

This function executes the core DMRG optimization, designed to perform stacking sequence retrieval with 
specific lamination parameters as targets. It accepts multiple initial MPS configurations in `psi0_list`, 
each used in a separate DMRG optimization trial within the same experiment, and saves results to an HDF5 file. 
The `sample_idx` parameter allows unique indexing for each experiment, which is helpful when running 
a series of experiments with different target lamination parameters. This index is added to the HDF5 filename.

Arguments:
  - `filelocation`: Directory where results will be saved.
  - `filename`: Base name for the HDF5 file.
  - `sample_idx`: Integer identifier for the experiment, used to distinguish different target parameters.
  - `laminate`: `Laminate` struct containing properties of the laminate.
  - `target_parameters`: Target matrix of lamination parameters, used to calculate optimization loss.
  - `constraints`: `ConstraintSettings` struct with manufacturing constraints.
  - `sites`: Vector of site indices defining the MPS sites.
  - `PH`: A `ProjMPOSum` object representing the sum of MPOs as the Hamiltonian for DMRG optimization.
  - `psi0_list`: Vector of initial MPS configurations, each used as a starting point for a separate DMRG trial within this experiment.
  - `num_sweeps`: Number of sweeps to perform during DMRG.
  - `sweep_sequence`: A string defining the order and direction of sweeps.
  - `max_bond_dim`: Maximum bond dimension for the MPS during DMRG.

Optional Arguments:
  - `save_psi0`: Boolean indicating whether to save the initial MPS configurations in the HDF5 file.
  - `save_properties`: Dictionary of additional metadata to store in the file, such as angles or constraint details.
  - `target_stack`: Optional stacking sequence corresponding to `target_parameters`. If provided, it will be saved in the file.
  - `kwargs`: Additional keyword arguments passed to `dmrg_custom` for fine-tuning DMRG parameters.

Usage Notes:
- `dmrg_experiment` can be called with various configurations:
  - `PH` can be a `ProjMPOSum`, `MPO`, or vector of `MPO` objects. If not provided, the MPO will be generated using `build_mpo`.
  - If only `target_stack` is specified, `target_parameters` will be inferred from it and included in the saved file.
  - By setting `sample_idx`, multiple DMRG experiments with different target parameters can be saved in separate HDF5 files.

Returns:
  - None. The function saves the results of each DMRG trial (including stacking sequences, lamination parameters, 
    energies, and constraint violations) to the specified HDF5 file.

Examples:
```julia
dmrg_experiment("results/", "experiment_data", 1, laminate, target_params, constraints,
                sites, PH, psi0_list, 10, "LRLR", 100, save_psi0=true)
```
"""
function dmrg_experiment(
    filelocation::String, filename::String, sample_idx::Int, laminate::Laminate,
    target_parameters::Matrix{Float64},
    constraints::ConstraintSettings,
    sites::Vector{<:Index}, PH::ProjMPOSum,psi0_list::Vector{MPS},
    num_sweeps::Int, sweep_sequence::String, max_bond_dim::Int;
    save_psi0::Bool = false,save_properties::Union{Dict{String,Any},Nothing}=nothing,
    target_stack::Union{Vector{Int},Nothing}=nothing,
    kwargs...
)

    # calculate energies for initial MPS
    psi0_energies = [energy_Hsum_psi(PH,psi0) for psi0 ∈ psi0_list]

    # create sweeps
    bond_dims = gen_bond_dims(num_sweeps,max_bond_dim)
    cutoffs = gen_cutoffs(num_sweeps)
    sweeps = Sweeps(num_sweeps)
    maxdim!(sweeps,bond_dims...)
    cutoff!(sweeps,cutoffs...)

    if target_stack !== nothing
        if save_properties === nothing
            save_properties = Dict{String,Any}()
        end
        save_properties["target_stack"] = target_stack
    end


    # create file
    filepath = create_file_for_dmrg_experiment(
        filelocation, filename, sample_idx,laminate,target_parameters,
        constraints,sweeps,sweep_sequence,psi0_energies,
        save_psi0 ? psi0_list : nothing;
        save_properties = save_properties
    )

    # perform optimizations
    for (t,psi0) ∈ enumerate(psi0_list)
        # output
        println("Try $t:")
        println("")

        # perform dmrg
        result:: DMRGResult = dmrg_custom(PH, psi0, sweeps, sweep_sequence,kwargs...)

        # get resulting stacking sequence, lamination parameters,
        # loss and constraint violations
        stack = result_psi_to_stack(result.psi)
        res_lp = parameters(laminate,stack)
        res_loss = sum((res_lp .- target_parameters).^2)
        constraint_violations = count_constraint_violations(stack,constraints)

        # print results
        println("Completed!")
        println("Last time:     $(result.times[end])")
        println("Last energy:   $(result.energies[end])")
        println("Loss:          $(res_loss)")
        println("RMSE:          $(sqrt(res_loss))")
        println("Constr. viol.: $(join([string(i) for i in constraint_violations], ", "))")
        println("✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳")
        println("")
        
        # Save results in HDF5
        fid = h5open(filepath,"r+")
        fid["data/energies"][t,:] = result.energies
        fid["data/time_stamps"][t,:] = result.times
        fid["data/elapsed_time_sweeps"][t,:] = result.sweep_durations
        fid["data/maxlinkdim"][t,:] = result.max_linkdims
        
        res_group = fid["results"]
        res_group["stack"][t,:] = stack
        res_group["lamination_parameters"][t,:] = vec(res_lp)
        res_group["loss"][t] = res_loss
        res_group["rmse"][t] = sqrt(res_loss)
        res_group["constraint_violations"][t,:] = collect(
            haskey(constraint_violations, sym) ?
            constraint_violations[sym] : 0
            for sym in ["nn","knearest","minimum","balanced"]
        )

        close(fid)
        println("Saved!")
        println("")

    end
end

function dmrg_experiment(
    filelocation::String, filename::String, sample_idx::Int, laminate::Laminate,
    target_stack::Vector{Int},
    constraints::ConstraintSettings,
    sites::Vector{<:Index}, PH::ProjMPOSum,psi0_list::Vector{MPS},
    num_sweeps::Int, sweep_sequence::String, max_bond_dim::Int;
    save_psi0::Bool = false,save_properties::Union{Dict{String,Any},Nothing}=nothing,
    kwargs...
)
    target_parameters = parameters(laminate,target_stack)
    
    return dmrg_experiment(
        filelocation, filename, sample_idx, laminate,
        target_parameters,
        constraints,
        sites, PH, psi0_list,
        num_sweeps, sweep_sequence, max_bond_dim,
        save_psi0 = save_psi0, save_properties=save_properties,
        target_stack = target_stack,
        kwargs...
    )
end

function dmrg_experiment(
    filelocation::String, filename::String, sample_idx::Int, laminate::Laminate,
    target_parameters::Matrix{Float64},target_stack::Vector{Int},
    constraints::ConstraintSettings,
    sites::Vector{<:Index}, PH::ProjMPOSum,psi0_list::Vector{MPS},
    num_sweeps::Int, sweep_sequence::String, max_bond_dim::Int;
    save_psi0::Bool = false,save_properties::Union{Dict{String,Any},Nothing}=nothing,
    kwargs...
)
    
    return dmrg_experiment(
        filelocation, filename, sample_idx, laminate,
        target_parameters,
        constraints,
        sites, PH, psi0_list,
        num_sweeps, sweep_sequence, max_bond_dim,
        save_psi0 = save_psi0, save_properties=save_properties,
        target_stack = target_stack,
        kwargs...
    )
end

# Define the symbols and types for positional and keyword arguments
arg_names = [(:target_parameters,), (:target_stack,), (:target_parameters, :target_stack)]
arg_types = [(Matrix{Float64},), (Vector{Int},), (Matrix{Float64}, Vector{Int})]
kwarg_names = [(:target_stack,), (), ()]
kwarg_types = [(Union{Vector{Int}, Nothing},), (), ()]

# Construct the argument list with types for positional arguments
arg_names_with_types = [
    [Expr(:(::), a, t) for (a, t) in zip(arg, typ)] 
    for (arg, typ) in zip(arg_names, arg_types)
]

# Construct the keyword argument list with types and default values
kwarg_names_with_types = [
    [Expr(:(=), Expr(:(::), a, t), :nothing) for (a, t) in zip(kwarg, kwtyp)] 
    for (kwarg, kwtyp) in zip(kwarg_names, kwarg_types)
]

# Dynamically define functions with both positional and keyword arguments
for (arg, arg_t, kwarg, kwarg_t) in zip(arg_names, arg_names_with_types, kwarg_names, kwarg_names_with_types)
    # println("Generating function...")

    eval(Meta.parse(
        """function dmrg_experiment(
            filelocation::String, filename::String, sample_idx::Int, laminate::Laminate,
            $(join(arg_t,",")),
            constraints::ConstraintSettings,
            sites::Vector{<:Index}, H_list::Vector{<:MPO}, psi0_list::Vector{MPS},
            num_sweeps::Int, sweep_sequence::String, max_bond_dim::Int;
            save_psi0::Bool = false, save_properties::Union{Dict{String,Any},Nothing}=nothing,
            $(join(["$kwt," for kwt in kwarg_t])) kwargs...
        )
            # Convert H_list
            H_list = permute.(H_list, Ref((linkind, siteinds, linkind)))
            PH = ProjMPOSum(H_list)
            
            # Call the original function with the converted args
            return dmrg_experiment(
                filelocation, filename, sample_idx, laminate,
                $(join(arg,",")), constraints, sites, PH, psi0_list,
                num_sweeps, sweep_sequence, max_bond_dim,
                save_psi0 = save_psi0, save_properties = save_properties,
                $(join(["$kw=$kw," for kw in kwarg]))
                kwargs...
            )
        end"""
    ))

    eval(Meta.parse(
        """function dmrg_experiment(
            filelocation::String, filename::String, sample_idx::Int, laminate::Laminate,
            $(join(arg_t,",")),
            constraints::ConstraintSettings,
            sites::Vector{<:Index}, H::MPO, psi0_list::Vector{MPS},
            num_sweeps::Int, sweep_sequence::String, max_bond_dim::Int;
            save_psi0::Bool = false, save_properties::Union{Dict{String,Any},Nothing}=nothing,
            $(join(["$kwt," for kwt in kwarg_t])) kwargs...
        )
            
            # Call the original function with the converted args
            return dmrg_experiment(
                filelocation, filename, sample_idx, laminate,
                $(join(arg,",")), constraints, sites, [H], psi0_list,
                num_sweeps, sweep_sequence, max_bond_dim,
                save_psi0 = save_psi0, save_properties = save_properties,
                $(join(["$kw=$kw," for kw in kwarg]))
                kwargs...
            )
        end"""
    ))

    eval(Meta.parse(
        """function dmrg_experiment(
            filelocation::String, filename::String, sample_idx::Int, laminate::Laminate,
            $(join(arg_t,",")),
            constraints::ConstraintSettings,
            sites::Vector{<:Index}, psi0_list::Vector{MPS},
            num_sweeps::Int, sweep_sequence::String, max_bond_dim::Int;
            save_psi0::Bool = false, save_properties::Union{Dict{String,Any},Nothing}=nothing,
            exclude_pars::Union{Vector{Tuple{Int, Int}}, Tuple{Int, Int}, Nothing} = nothing,
            $(join(["$kwt," for kwt in kwarg_t])) kwargs...
        )
            $(
                :target_parameters in arg ? "" :
                "target_parameters = parameters(laminate,target_stack)\n\n" 
            )
            # build MPO
            mpo = build_mpo(sites,laminate,target_parameters,constraints,exclude_pars=exclude_pars)
            
            # Call the original function with the converted args
            return dmrg_experiment(
                filelocation, filename, sample_idx, laminate,
                $(join(arg,",")), constraints, sites, mpo, psi0_list,
                num_sweeps, sweep_sequence, max_bond_dim,
                save_psi0 = save_psi0, save_properties = save_properties,
                $(join(["$kw=$kw," for kw in kwarg]))
                kwargs...
            )
        end"""
    ))
end
