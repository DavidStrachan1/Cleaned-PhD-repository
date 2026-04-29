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
using NDTensors: unwrap_array_type
using DataFrames

include("Helper_functions.jl")
include("Evolution_and_analysis_functions.jl")
include("HamiltonianBuilding.jl")
include("ITensor_functions.jl")
include("Mpemba_functions.jl")
include("Initialisation.jl")


end