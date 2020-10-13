
using JLD2, StaticArrays, DelimitedFiles

fpath = "/home/stc/devspace/codes/SpinEchoSim/"

# CPU, liouville part not cleaned up yet
include(join([fpath,"SpinEchoSim_cpu.jl"]))

#GPU
#using CUDA
#include(join([fpath,"SpinEchoSim_gpu.jl"]))


# setup the job
# interaction
α = [0.01];

# number of frequencies
n = (50, 50)

# make the parameter file
params = make_params();
params["α"] = α;
params["n"] = n;

# make a lattice, pbc = periodic bc or not
hlk = [1; 1]
θ = [π/2]
pbc = true;
r, spin_idx = make_lattice(hlk, θ, n);
params["spin_idx"] = spin_idx
params["pbc"] = pbc

# make the stencil
# ξ = correlation length, (hlk, θ, n, r) = lattice parameters, pbc = periodic bc or not, p = decay pow (1/r^p)
# for sparse: t1 = threshold for starting decay, t2 = cutoff to go to zero, cutoff = exp(-(d_coeff*r)^d_pow)
t1 = 0.1
t2 = 0.01
d_pow = 2
d_coef = 0.75
ξ = 10;
p = 4;
stencil = make_stencil(r, ξ, p);
# s_stencil = make_sparse_stencil(r, ξ, n, t1, t2, d_pow, d_coef)
params["M_stencil"] = stencil;
params["ξ"] = ξ

# load dissipation parameters
Γ = (0, 0, 10^-3);
params["Γ"] = Γ

# load the pulsing parameters
# f = collect(LinRange(0.1, 2, 11))
ϕ = (0, π/2)
Ix = convert.(Complex{Float32}, @SArray [0 1/2; 1/2 0])
Iy = convert.(Complex{Float32}, @SArray [0 -1im/2; 1im/2 0]);

coef90 = Ix*cos(ϕ[1]) + Iy*sin(ϕ[1])
coef180 = Ix*cos(ϕ[2]) + Iy*sin(ϕ[2])
U90 = convert.(Complex{Float32}, exp(-1im*pi*coef90/2));
U180 = convert.(Complex{Float32}, exp(-1im*pi*coef180));
    
params["U90"] = U90
params["U180"] = U180
# params["f"] = f;

# load τ list
# τ = exp10.(LinRange(-5, log10(600e-6), 10));
# params["τ"] = τ;

# cpmg parameters
#echo_time = collect(LinRange(50e-6, 300e-6, 20));
#num_echoes = unique(convert.(Int64, round.(1500e-6./echo_time)));
#echo_time = round.(round.(1500 ./num_echoes)*1e-6, digits = 6);
#params["echo_time"] = echo_time;
#params["num_echoes"] = num_echoes;
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

	stencil = make_stencil(r, ξ, p);
	# s_stencil = make_sparse_stencil(r, ξ, n, t1, t2, d_pow, d_coef)
        Γ = (0, 0, 10^(-d));
        params["α"] = [α];
	params["M_stencil"] = stencil;
	params["ξ"] = ξ
        params["Γ"] = Γ

	I, d = make_idx(vars, params)
	# generate temporary parameters
	tparams = make_temp_params(params, vars, I[1])

	# simulate
	#M_list[idx] = spin_echo_sim(tparams)
	M_list[idx] = spin_echo_sim_liouville(tparams)
end

fname = "echos.jld2";
@save fname M_list
