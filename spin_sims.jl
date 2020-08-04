include("liouville_tools.jl")
using .LiouvilleTools
using LinearAlgebra

function spin_echo_sim(params)
    
    # initialize M_list
    M_list = [];
    
    U90 = params["U90"];
    
    # 90 pulse
    ψ_list = [U90*ψ for ψ in params["ψ_init"]];
    
    # first tau
    t0 = 0.0;
    ψ_list, M_list, t1 = time_propagate(ψ_list, M_list, t0, params["dt"], params["nτ"], params)
    
    # 180 pulse
    ψ_list = [U90*U90*ψ for ψ in ψ_list];
        
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
    
    # initial time
    t = t0;

    # initial magnetization    
    M_eval = [tr(M_op*ψ*ψ') for ψ in ψ_list]
    M = sum(P.*M_eval);
    
    # see if we are doing a local stencil for Magnetization
    if params["local_M_on"]
        M_stencil = params["M_stencil"]
        M_stencil_vec = [P.*shift_stencil(M_stencil, vec, params["n"], params["pbc"]) for vec in params["spin_idx"]]
        M_stencil_vec = [M_stencil_vec[j]/sum(M_stencil_vec[j]) for j = 1:nS] # normalize
        M_local = [sum(M_stencil_vec[j].*M_eval ) for j = 1:nS]
    end
    
    # time evolve
    for idx = 1:nsteps
        
        t += dt;
        
        if params["local_M_on"]
            int = get_int(t, M_local, params)
            U = [exp(-1im*( -(ν[j]-ν0)*Iz - int[j])*dt) for j = 1:nS];
        else
            int = get_int(t, M, params)
            U = [exp(-1im*( -(ν[j]-ν0)*Iz - int[j])*dt) for j = 1:nS];
        end
        
        # time evolve
        ψ_list = [U[j]*ψ_list[j] for j = 1:nS];
        
        # update M and save value
        M_eval = [tr(M_op*ψ*ψ') for ψ in ψ_list]
        M = sum(P.*M_eval);
        push!(M_list, M);

        # local Magnetization update
        if params["local_M_on"]
            M_local = [sum(M_stencil_vec[j].*M_eval ) for j = 1:nS]
        end
                
    end
    
    return ψ_list, M_list, t
    
end

function get_int(t, M, params)
    
    α = params["α"]
    ω = params["ω"]
    k = params["k"]
    r = params["r"]
    
    if params["local_M_on"]
        int = [α*cos(ω*t + dot(k, r[j]))*getOprime(t, M[j], params) for j = 1:params["nfreq"]]
    else
        Oprime = getOprime(t, M, params)
        int = [α*cos(ω*t + dot(k, r[j]))*Oprime for j = 1:params["nfreq"]]
    end
    
    return int

end
    

function getOprime(t, M, params)
    
    # interaction parameters
    ν0 = params["ν0"]
    
    # Oprime = (1im/2)*[0 -exp(-1im*ν0*t); exp(1im*ν0*t) 0]; # Iy
    Oprime = (1/4)*[0 conj(M); M 0] - (1/4)*[0 M*exp(-2im*ν0*t); conj(M)*exp(2im*ν0*t) 0]; # IyMy
    # Oprime = (1im/4)*[0 -conj(M); M 0] + (1im/4)*[0 -M*exp(-2im*ν0*t); conj(M)*exp(2im*ν0*t) 0]; # IyMx
    # Oprime = (1im/4)*[0 conj(M); -M 0] + (1im/4)*[0 -M*exp(-2im*ν0*t); conj(M)*exp(2im*ν0*t) 0]; # IxMy
    # Oprime = (1/4)*[0 conj(M); M 0] + (1/4)*[0 M*exp(-2im*ν0*t); conj(M)*exp(2im*ν0*t) 0]; #IxMx
    
    return Oprime
    
end

function spin_echo_sim_liouville(params)   
    
    # initialize M_list
    M_list = [];
    
    UL90 = params["UL90"]
    UR90 = params["UR90"]    
    
    # 90 pulse
    ρ_list_L = [dm_H2L(UL90*ρ*UR90) for ρ in params["ρ_init"]];    
    
    # first tau
    t0 = 0.0;
    ρ_list_L, M_list, t1 = time_propagate_liouville(ρ_list_L, M_list, t0, params["dt"], params["nτ"], params)
    
    # 180 pulse
    ρ_list_L = [dm_H2L(UL90*UL90* dm_L2H(ρ_L) *UR90*UR90) for ρ_L in ρ_list_L];
        
    # second tau
    ρ_list_L, M_list, t2 = time_propagate_liouville(ρ_list_L, M_list, t1, params["dt"], 2*params["nτ"], params)
    
  return M_list

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
    
    # interaction parameters
    α = params["α"]
    ω = params["ω"]
    
    # initial time
    t = t0;

    # jump operators
    Lj_list = params["Lj"]
    J_L = JumpsToSuper(Lj_list) # get dissipative super operator (assumed constant in time)
  
    # initial magnetization    
    M_L = leftop_H2L(M_op)
    M_eval = [tr_L(M_L*ρ_list_L[j]) for j = 1:nS]
    M = sum(P.*M_eval);
    
    # see if we are doing a local stencil for Magnetization
    if params["local_M_on"]
        M_stencil = params["M_stencil"]
        M_stencil_vec = [P.*M_stencil_shift(M_stencil,j) for j = 1:nS]
        M_stencil_vec = [M_stencil_vec[j]/sum(M_stencil_vec[j]) for j = 1:nS] # normalize ?
        M_local = [sum(M_stencil_list[j].*M_eval ) for j = 1:nS]
    end    
    
    for idx = 1:nsteps
        
        t += dt;
        
        
        if params["local_M_on"]
            Oprime_local = [getOprime(t, M_local[j], params) for j = 1:nS]
            Ham_L = [HamToSuper( -(ν[j]-ν0)*Iz - α*cos(ω*t + dot(k, r[j,:]))*Oprime_local[j] ) for j = 1:nS]
        else
            Oprime = getOprime(t, M, params)
            Ham_L = [HamToSuper( -(ν[j]-ν0)*Iz - α*cos(ω*t + dot(k, r[j,:]))*Oprime ) for j = 1:nS]
        end                
        
        U_L = [exp((-1im*Ham_L[j] + J_L )*dt) for j = 1:nS]
        
        # time evolve
        ρ_list_L = [U_L[j]*ρ_list_L[j] for j = 1:nS]
        
        # update M and save value
        M_eval = [tr_L(M_L*ρ_list_L[j]) for j = 1:nS]
        M = sum(P.*M_eval);
        push!(M_list, M)

        # local Magnetization update
        if params["local_M"]
            M_local = [sum(M_stencil_vec[j].*M_eval ) for j = 1:nS]
        end
        
    end
        
    return ρ_list_L, M_list, t
    
end