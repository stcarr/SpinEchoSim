## ONLY WORKS FOR BOTH DISSIPATION AND LOCAL M

include("../lib/liouville_tools.jl")

using .LiouvilleTools
using LinearAlgebra

function spin_echo_sim(params)
    
    # initialize M_list
    M_list = [];
    
    U90 = params["U90"];
    U180 = params["U180"];
    
    # 90 pulse
    ψ_list = [U90*ψ for ψ in params["ψ_init"]];
    
    # first tau
    t0 = 0.0;
    ψ_list, M_list, t1 = time_propagate(ψ_list, M_list, t0, params["dt"], params["nτ"], params)
    
    # 180 pulse
    ψ_list = [U180*ψ for ψ in ψ_list];
        
    # second tau
    ψ_list, M_list, t2 = time_propagate(ψ_list, M_list, t1, params["dt"], 2*params["nτ"], params)
    
  return M_list

end

function time_propagate(ψ_list, M_list, t0, dt, nsteps, params)
        
    # spectrum info
    ν0 = params["ν0"] # central freq.
    ν = params["ν"] # spin freqs.
    P = params["P"] # spin weights
    nS = params["nfreq"] # number of spins
    r = params["r"] # positions of spins
    
    # operators
    M_op = params["M_op"]
    Iz = params["Iz"]
    
    # additional values
    n = params["n"]
    spin_idx = params["spin_idx"]
    
    # experiment parameters
    α = params["α"]
    
    # initial time
    t = t0;

    # initial magnetization    
    M_eval = [tr(M_op*ψ*ψ') for ψ in ψ_list]
    M = sum(P.*M_eval);
    
    # prepare the stencils
    M_stencil = params["M_stencil"]
    M_stencil_vec = shift_stencil(M_stencil, P, spin_idx, n)

    # calculate local M
    M_local = [sum(M_eval.*M_stencil_vec[:,:,v[1],v[2]]) for v in spin_idx]
    
    # time evolve
    for idx = 1:nsteps
        
        t += dt;

        # calculate interaction
        Oprime = map(x -> (1/4)*[0 conj(x); x 0] - (1/4)*[0 x*exp(-2im*ν0*t); conj(x)*exp(2im*ν0*t) 0], M_local)
        int = α*Oprime

        # calculate propagators
        H = -(ν.-ν0).*[Iz] .- int
        U = map(exp, -1im*H*dt)
        
        # time evolve
        ψ_list = U.*ψ_list;
        
        # calculate local M
        M_eval = [tr(M_op*ψ*ψ') for ψ in ψ_list]
        
        # calculate global M and save
        M = sum(P.*M_eval);
        push!(M_list, M);
        
        # calculate local M
        M_local = [sum(M_eval.*M_stencil_vec[:,:,v[1],v[2]]) for v in spin_idx]
        
    end
    
    return ψ_list, M_list, t
    
end

function spin_echo_sim_liouville(params)   
    
    # initialize M_list
    M_list = [];
    
    UL90 = params["U90"];
    UR90 = UL90'
    
    UL180 = params["U180"];
    UR180 = UL180'
    
    # 90 pulse
    ψ_list = [UL90*ψ for ψ in params["ψ_init"]];
    
    # convert to liouville space
    ρ_list_L = [dm_H2L(ψ*ψ') for ψ in ψ_list];    
    
    # first tau
    t0 = 0.0;
    ρ_list_L, M_list, t1 = time_propagate_liouville(ρ_list_L, M_list, t0, params["dt"], params["nτ"], params)
    
    # 180 pulse
    ρ_list_L = [dm_H2L(UL180* dm_L2H(ρ_L) *UR180) for ρ_L in ρ_list_L];
        
    # second tau
    ρ_list_L, M_list, t2 = time_propagate_liouville(ρ_list_L, M_list, t1, params["dt"], 2*params["nτ"], params)
    
  return M_list

end

function spin_echo_sim_liouville_cpmg(params)   
    
    # initialize M_list
    M_list = [];
    
    UL90 = params["U90"];
    UR90 = UL90'
    
    UL180 = params["U180"];
    UR180 = UL180'
    
    # 90 pulse
    ψ_list = [UL90*ψ for ψ in params["ψ_init"]];
    
    # convert to liouville space
    ρ_list_L = [dm_H2L(ψ*ψ') for ψ in ψ_list];
    
    # initial time
    t0 = 0.0;
    nτ = convert(Int64, round(params["echo_time"]*params["γ"]/params["dt"]));
    
    # an array which holds the echoes individually
    echoes = Array{Any}(undef, params["num_echoes"], nτ);
    
    # fid
    ρ_list_L, M_list, t1 = time_propagate_liouville(ρ_list_L, M_list, t0, params["dt"], convert(Int64, round(nτ/2)), params)

    # update t0
    t0 = t1;
    
    ρ_list_L = [dm_H2L(UL180* dm_L2H(ρ_L) *UR180) for ρ_L in ρ_list_L];
    
    for echo_idx = 1:params["num_echoes"]
    
        # echo
        ρ_list_L, M_list, t1 = time_propagate_liouville(ρ_list_L, M_list, t0, params["dt"], nτ, params)

        # 180 pulse
        ρ_list_L = [dm_H2L(UL180* dm_L2H(ρ_L) *UR180) for ρ_L in ρ_list_L];
        
        # update time
        t0 = t1;
        
        # save echo
        t0_idx = convert(Int64, round(nτ/2)) + (echo_idx-1)*nτ + 1
        tf_idx = convert(Int64, round(nτ/2)) + echo_idx*nτ
        echoes[echo_idx,:] = M_list[t0_idx:tf_idx];

    end
    
  return M_list, echoes

end

function time_propagate_liouville(ρ_list_L, M_list, t0, dt, nsteps, params) 
            
    # spectrum info
    ν0 = params["ν0"] # central freq.
    ν = params["ν"] # spin freqs.
    P = params["P"] # spin weights
    nS = params["nfreq"] # number of spins
    
    # operators
    M_op = params["M_op"]
    Iz = params["Iz"]
    
    # additional values
    n = params["n"]
    spin_idx = params["spin_idx"]
    
    # interaction parameters
    α = params["α"]
    
    # initial time
    t = t0;

    # jump operators
    Lj_list = params["Lj"]
    J_L = JumpsToSuper(Lj_list) # get dissipative super operator (assumed constant in time)
  
    # initial magnetization    
    M_L = leftop_H2L(M_op)
    M_eval = [tr_L(M_L*ρ) for ρ in ρ_list_L]
    M = sum(P.*M_eval);

    # prepare the stencils
    M_stencil = params["M_stencil"]
    M_stencil_vec = shift_stencil(M_stencil, P, spin_idx, n)

    # calculate local M
    M_local = [sum(M_eval.*M_stencil_vec[:,:,v[1],v[2]]) for v in spin_idx]
    
    # time evolve
    for idx = 1:nsteps
        
        t += dt;
        
        # calculate interaction
        int = α*map(x -> (1/4)*[0 conj(x); x 0] - (1/4)*[0 x*exp(-2im*ν0*t); conj(x)*exp(2im*ν0*t) 0], M_local)
        
        # calculate hamiltonian
        H_H = -(ν.-ν0).*[Iz] .- int
        H_L = [HamToSuper(H) for H in H_H]
        
        # calculate propagators, adding in dissipation
        U_L = [exp(( -1im*H + J_L )*dt) for H in H_L]
        
        # time evolve
        ρ_list_L = U_L.*ρ_list_L
        
        # update M and save value
        M_eval = [tr_L(M_L*ρ) for ρ in ρ_list_L]
        M = sum(P.*M_eval);
        push!(M_list, M)

        # calculate local M
        M_local = [sum(M_eval.*M_stencil_vec[:,:,v[1],v[2]]) for v in spin_idx]
        
    end
        
    return ρ_list_L, M_list, t
    
end


function getOprime(t, M, params)
    
    # interaction parameters
    ν0 = params["ν0"]
    
    # Oprime = (1im/2)*[0 -exp(-1im*ν0*t); exp(1im*ν0*t) 0]; # Iy
    Oprime = [(1/4)*[0 conj(x); x 0] - (1/4)*[0 x*exp(-2im*ν0*t); conj(x)*exp(2im*ν0*t) 0] for x in M]; # IyMy
    # Oprime = (1im/4)*[0 -conj(M); M 0] + (1im/4)*[0 -M*exp(-2im*ν0*t); conj(M)*exp(2im*ν0*t) 0]; # IyMx
    # Oprime = (1im/4)*[0 conj(M); -M 0] + (1im/4)*[0 -M*exp(-2im*ν0*t); conj(M)*exp(2im*ν0*t) 0]; # IxMy
    # Oprime = (1/4)*[0 conj(M); M 0] + (1/4)*[0 M*exp(-2im*ν0*t); conj(M)*exp(2im*ν0*t) 0]; #IxMx
    
    return Oprime
    
end

## CHECK FOR PERIODIC BOUNDARY CONDITIONS AND DO THE SHIFT
function shift_stencil(stencil, P, spin_idx, n)

    bigS = zeros(n[1], n[2], n[1], n[2])

    for v in spin_idx
        shift = (spin_idx[v...][1] - 1, spin_idx[v...][2] - 1)
        temp = P.*circshift(stencil, shift)
        bigS[:,:,v[1],v[2]] = temp/sum(temp)
    end

    return bigS

end   
