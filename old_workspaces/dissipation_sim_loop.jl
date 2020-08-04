using Plots
using StatsBase
using LinearAlgebra
using Statistics
using JLD2
using Dates
using LsqFit
using StaticArrays

# data structure module (needs to be module to avoid overwriting local variables in REPL scope)
include("SpinSimParams.jl")
using .SpinSimParams

# function libraries
include("spin_sims.jl");

fprefix = "looprun_06-16-2020"

# gamma
γ = 2*π*1e6;

# time variables
τ = 100e-6;
dt = 2;

# interaction
α = LinRange(0,0.2,21)
ω = exp10.(LinRange(-3,-1,8+1))
pushfirst!(ω,0)
# spin ensemble
ν0 = 10;
bw = 0.5;
dfreq = 0.002;

# initial conditions
dim = 2;
ψ_0 = @SArray [1 0];
ρ_temp = [ψ_0[i]*ψ_0[j] for i = 1:dim, j = 1:dim];

ρ0 = @SMatrix [ρ_temp[1,1] ρ_temp[1,2]; ρ_temp[2,1] ρ_temp[2,2]];

params = make_parameters(γ, τ, dt, α, ω, ν0, bw, dfreq, ρ0);

Lj_array = []

gam_0_list = exp10.(LinRange(-6,-2, 4*2 + 1))

for idx = 1:size(gam_0_list,1)
    gam_0 = gam_0_list[idx]
    gam_1 = gam_0
    gam_2 = gam_0
    gam_3 = 0.0

    Iz = @SMatrix [1/2 0; 0 -1/2];

    nJ = 3
    Lj_list = []
    push!(Lj_list, sqrt(gam_1)* @SMatrix [0 1; 0 0]) # down to up flip
    push!(Lj_list, sqrt(gam_2)* @SMatrix [0 0; 1 0]) # up to down flip
    push!(Lj_list, sqrt(gam_3)*Iz) # phase decoherence
    push!(Lj_array,Lj_list)
end

params.Lj = Lj_array
vars = ["α", "ω", "Lj"];


I, d = make_idx(vars, params)
M_list = Array{Any}(undef, d)

for i in I
    println(i) # print loop index for gauging speed locally
    tparams = temp_parameters(params, vars, i)
    M_list[i] = spin_echo_sim_liouville(tparams)
end

@save string(fprefix,"_data.jld2") M_list params