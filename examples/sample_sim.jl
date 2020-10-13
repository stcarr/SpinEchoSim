

using Plots, StatsBase, LinearAlgebra, 
      Statistics, JLD2, Dates,
      StaticArrays, JSON
# CPU, liouville part not cleaned up yet
include("../SpinEchoSim_cpu.jl")

#GPU
#using CUDA
#include("../SpinEchoSim_gpu.jl")


# setup the job
# interaction
α = [0.01];

# number of frequencies
n = (75, 75)

# make the parameter file
params = make_params(α, n);

# make a lattice, pbc = periodic bc or not
hlk = [1; 1]
θ = [π/2]
pbc = true;
r, spin_idx = make_lattice(hlk, θ, n, pbc);
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
stencil = make_stencil(hlk, θ, n, r, ξ, pbc, p);
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


α = 0.1; # correlation sterngth
ξ = 10; # correlation length
p = 4; # power
stencil = make_stencil(hlk, θ, n, r, ξ, pbc, p);
# s_stencil = make_sparse_stencil(r, ξ, n, t1, t2, d_pow, d_coef)
params["α"] = [α];
params["M_stencil"] = stencil;
params["ξ"] = ξ

I, d = make_idx(vars, params)
# generate temporary parameters
tparams = make_temp_params(params, vars, I[1])

# simulate
M_list[i] = spin_echo_sim(tparams)
#M_list[i] = spin_echo_sim_liouville(tparams)

fname = "spinecho_M.jld2";
@save fname M_list




