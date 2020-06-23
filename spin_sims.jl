include("liouville_tools.jl")
using .LiouvilleTools
using LinearAlgebra

function spin_echo_sim(params)
    
    # initialize M_list
    M_list = [];
    
    UL90 = params["UL90"];
    UR90 = params["UR90"];
    
    # 90 pulse
    ρ_list = [UL90*ρ*UR90 for ρ in params["ρ_init"]];
    
    # first tau
    t0 = 0.0;
    ρ_list, M_list, t1 = time_propagate(ρ_list, M_list, t0, params["dt"], params["nτ"], params)
    
    # 180 pulse
    ρ_list = [UL90*UL90*ρ*UR90*UR90 for ρ in ρ_list];
        
    # second tau
    ρ_list, M_list, t2 = time_propagate(ρ_list, M_list, t1, params["dt"], 2*params["nτ"], params)
    
  return M_list

end

function time_propagate(ρ_list, M_list, t0, dt, nsteps, params)
        
    # spectrum info
    ν0 = params["ν0"] # central freq.
    ν = params["ν"] # spin freqs.
    P = params["P"] # spin weights
    nS = params["nfreq"] # number of spins
    r = params["r"] # positions of spins
    
    # operators
    M_op = params["M_op"]
    Iz = params["Iz"]
    
    # interaction parameters
    α = params["α"]
    ω = params["ω"]
    k = params["k"]
    
    # initial time
    t = t0;

    # initial magnetization
    M = sum([P[j].*tr(M_op*ρ_list[j]) for j = 1:nS]);
    
    # time evolve
    for idx = 1:nsteps
        
        t += dt;
        
        Oprime = getOprime(t, M, params)

        UL = [exp(-1im*( -(ν[j]-ν0)*Iz - α*cos(ω*t + dot(k, r[j,:]))*Oprime )*dt) for j = 1:nS];
        UR = [exp( 1im*( -(ν[j]-ν0)*Iz - α*cos(ω*t + dot(k, r[j,:]))*Oprime )*dt) for j = 1:nS];        
        
        # time evolve
        ρ_list = [UL[j]*ρ_list[j]*UR[j] for j = 1:nS];
        
        # update M and save value
        M = sum([P[j]*tr(M_op*ρ_list[j]) for j = 1:nS]);
        push!(M_list, M);
        
    end
    
    return ρ_list, M_list, t
    
end

function getOprime(t, M, params)
   
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
    M = sum([P[j].*tr_L(M_L*ρ_list_L[j]) for j = 1:nS])
    
    for idx = 1:nsteps
        
        t = t + dt;
        
        Oprime = getOprime(t, M, params)
        
        Ham_L = [HamToSuper( -(ν[j]-ν0)*Iz - α*cos(ω*t + dot(k, r[j,:]))*Oprime ) for j = 1:nS]

        U_L = [exp((-1im*Ham_L[j] + J_L )*dt) for j = 1:nS]      
        
        # time evolve
        ρ_list_L = [U_L[j]*ρ_list_L[j] for j = 1:nS]
        
        # update M and save value
        M = sum([P[j].*tr_L(M_L*ρ_list_L[j]) for j = 1:nS])
        push!(M_list, M)
        
    end
        
    return ρ_list_L, M_list, t
    
end