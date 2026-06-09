module PhDProject

using ITensors
using LinearAlgebra
using PolyChaos
using Trapz
using DSP
using Plots 
using ProgressMeter
using ITensorMPS
using Observers
using ProgressBars
using Distributed 
using SharedArrays
using LaTeXStrings
using Kronecker
using SparseArrays
using Base.Threads
using Adapt: adapt
using ITensors:
      ITensors,
      Algorithm,
      Index,
      ITensor,
      @Algorithm_str,
      δ,
      commonind,
      dag,
      denseblocks,
      directsum,
      hasqns,
      prime,
      scalartype,
      uniqueinds
using NDTensors: unwrap_array_type
using DataFrames

include("PhD_modules\\Helper_functions.jl")
include("PhD_modules\\Evolution_and_analysis_functions.jl")
include("PhD_modules\\HamiltonianBuilding.jl")
include("PhD_modules\\ITensor_functions.jl")
include("PhD_modules\\Mpemba_functions.jl")
include("PhD_modules\\Initialisation.jl")



end