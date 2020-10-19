

using Plots, StatsBase, LinearAlgebra, 
      Statistics, JLD2, Dates,
      StaticArrays, JSON

# CPU, liouville part not cleaned up yet
include("../SpinEchoSim_cpu.jl")

# GPU
# using CUDA
# include("../SpinEchoSim_gpu.jl")

### setup the job

# make the parameter file
params = make_params()

# interaction
params["α"] = 0.03;

# number of frequencies
params["n"] = (50, 50)

# make a lattice, pbc = periodic bc or not
params["hlk"] = [1; 1]
params["θ"] = [π/2]
params["r"], params["spin_idx"] = make_lattice(params["hlk"], params["θ"], params["n"])

# make the stencil
params["ξ"] = 10
params["decay_power"] = 4
params["M_stencil"] = make_stencil(params["r"], params["ξ"], params["decay_power"])

# dissipation parameters
params["Γ"] = (0, 0, 10^-3);

# load the pulsing parameters
params["flip_angle"] = π/2
params["phases"] = (0, π/2)

# cpmg parameters
params["echo_time"] = 100e-6 # collect(LinRange(50e-6, 300e-6, 20));
params["num_echoes"] = convert.(Int64, round.(1000e-6./params["echo_time"]));

# set the variables
vars = ["α"]
params["vars"] = vars;

# generate indexing (needed for temp_params function)
I, d = make_idx(vars, params)

# generate temporary parameters
tparams = make_temp_params(params, vars, I[1])

# simulate
M = spin_echo_sim_liouville(tparams)

fname = "spinecho_M.jld2";
@save fname M




