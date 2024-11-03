"""
    SweepNextToRight

Variant of `SweepNext` (in ITensors.jl: src/mps/sweeps.jl), but explicitly used for
right-ward sweeps in `Base.iterate`.
"""
struct SweepNextToRight
  N::Int
  ncenter::Int
end


"""
    sweepnext_to_right(N::Int; ncenter::Int=2)::SweepNextToRight

Variant of `sweepnext(N::Int; ncenter::Int=2)` (in ITensors.jl: src/mps/sweeps.jl),
but returns a SweepNextToRight
"""
function sweepnext_to_right(N::Int; ncenter::Int=2)::SweepNextToRight
  if ncenter < 0
    error("ncenter must be non-negative")
  end
  return SweepNextToRight(N, ncenter)
end


"""
    Base.iterate(sn::SweepNextToRight, state=(0, 1))

Variant of `Base.iterate(sn::SweepNext, state=(0, 1))` 
(in ITensors.jl: src/mps/sweeps.jl), but only performing a right-ward sweep.

Produces states (1, 1), (2, 1), ..., (N-1, 1)
"""
function Base.iterate(sn::SweepNextToRight, state=(0, 1))
  b, ha = state
  bstop = sn.N - sn.ncenter + 2
  new_b = b + 1
  new_ha = ha
  done = false
  if new_b == bstop
    return nothing
  end
  return ((new_b, new_ha), (new_b, new_ha))
end

"""
    SweepNextToLeft

Variant of `SweepNext` (in ITensors.jl: src/mps/sweeps.jl), but explicitly used for
left-ward sweeps in `Base.iterate`.
"""
struct SweepNextToLeft
  N::Int
  ncenter::Int
end

"""
    sweepnext_to_left(N::Int; ncenter::Int=2)::SweepNextToLeft

Variant of `sweepnext(N::Int; ncenter::Int=2)` (in ITensors.jl: src/mps/sweeps.jl),
but returns a SweepNextToLeft
"""
function sweepnext_to_left(N::Int; ncenter::Int=2)::SweepNextToLeft
  if ncenter < 0
    error("ncenter must be non-negative")
  end
  return SweepNextToLeft(N, ncenter)
end

"""
    Base.iterate(sn::SweepNextToLeft, state=(0, 1))

Variant of `Base.iterate(sn::SweepNext, state=(0, 1))` 
(in ITensors.jl: src/mps/sweeps.jl), but only performing a left-ward sweep

Produces states (N-1, 2),(N-2, 2), ..., (1, 2)
"""
function Base.iterate(sn::SweepNextToLeft, state=(-1, 2))
  b, ha = state
  if b == -1
    b = sn.N - sn.ncenter + 2
  end
  bstop = 0
  new_b = b - 1
  new_ha = ha
  done = false
  if new_b == bstop
    return nothing
  end
  return ((new_b, new_ha), (new_b, new_ha))
end