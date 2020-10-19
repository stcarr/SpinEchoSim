
using StaticArrays, DelimitedFiles

fpath = "/users/scarr8/codes/SpinEchoSim/"

# CPU, liouville part not cleaned up yet
#include(join([fpath,"SpinEchoSim_cpu.jl"]))

# GPU
using CUDA
include(join([fpath,"SpinEchoSim_gpu.jl"]))

# setup the job

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

fname = "inputs.txt";
input_params = readdlm(fname, Float64)
num_samps = size(input_params)[1]

M_list = Array{Any}(undef, num_samps)
for idx in range(1,length=num_samps)

	α = input_params[idx,1]; # correlation sterngth
	ξ = input_params[idx,2]; # correlation length
	p = input_params[idx,3]; # correlation power
    d = input_params[idx,4]; # dissipation power

	print("starting echo ",string(idx),"/",string(num_samps),"\n")

    Γ = (0, 0, 10^(-d));
    params["α"] = [α];
	params["M_stencil"] = make_stencil(r, ξ, p);
	params["ξ"] = ξ
    params["Γ"] = Γ

    # generate indexing for temp_params
	I, d = make_idx(vars, params)
    
	# generate temporary parameters
	tparams = make_temp_params(params, vars, I[1])

	# simulate
	M_list[idx] = spin_echo_sim_liouville(tparams)
    
end

fname_r = "echos_r.txt"
fname_i = "echos_i.txt";
open(fname_r, "w") do io
	writedlm(io, real(M_list))
end
open(fname_i, "w") do io
	writedlm(io, imag(M_list))
end

