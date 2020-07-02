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
    M_eval = [tr(M_op*ρ_list[j]) for j = 1:nS]
    M = sum(P.*M_eval);
    
    # see if we are doing a local stencil for Magnetization
    if params["local_M_on"]
        M_stencil = params["M_stencil"]
        M_stencil_vec = [P.*M_stencil_shift(M_stencil,j) for j = 1:nS]
        M_stencil_vec = [M_stencil_vec[j]/sum(M_stencil_vec[j]) for j = 1:nS] # normalize ?
        M_local = [sum(M_stencil_vec[j].*M_eval ) for j = 1:nS]
    end

    # time evolve
    for idx = 1:nsteps
        
        t += dt;
        
        if params["local_M_on"]
            Oprime_local = [getOprime(t, M_local[j], params) for j = 1:nS]
            UL = [exp(-1im*( -(ν[j]-ν0)*Iz - α*cos(ω*t + dot(k, r[j,:]))*Oprime_local[j] )*dt) for j = 1:nS];
            UR = [exp( 1im*( -(ν[j]-ν0)*Iz - α*cos(ω*t + dot(k, r[j,:]))*Oprime_local[j] )*dt) for j = 1:nS];
        else
            Oprime = getOprime(t, M, params)
            UL = [exp(-1im*( -(ν[j]-ν0)*Iz - α*cos(ω*t + dot(k, r[j,:]))*Oprime )*dt) for j = 1:nS];
            UR = [exp( 1im*( -(ν[j]-ν0)*Iz - α*cos(ω*t + dot(k, r[j,:]))*Oprime )*dt) for j = 1:nS];
        end
        
        # time evolve
        ρ_list = [UL[j]*ρ_list[j]*UR[j] for j = 1:nS];
        
        # update M and save value
        M_eval = [tr(M_op*ρ_list[j]) for j = 1:nS]
        M = sum(P.*M_eval);
        push!(M_list, M);

        # local Magnetization update
        if params["local_M_on"]
            M_local = [sum(M_stencil_vec[j].*M_eval ) for j = 1:nS]
        end
                
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

# changes origin of the "M_stencil", which computes a local <M> (instead of global) for Oprime
function M_stencil_shift(M_stencil, spin_idx)
    
    nx = size(M_stencil,1)
    ny = size(M_stencil,2)
    nz = size(M_stencil,3)
    
    # find the [x,y,z] coordinate for the givin spin index
    tar_loc_mat = zeros(nx*ny*nz,1)
    tar_loc_mat[spin_idx] = 1
    tar_loc_mat = reshape(tar_loc_mat, (nx,ny,nz))
    target_vec = (findall(x->x==1, tar_loc_mat)[1])
    
    # turn that coordinate into an array, with -1 for one-based indexing (e.g. no shift if at [1,1,1])
    shift_vec = zeros(3)
    shift_vec[1] = target_vec[1]-1
    shift_vec[2] = target_vec[2]-1
    shift_vec[3] = target_vec[3]-1

    # move the stencil to the spin center
    M_new = circshift(M_stencil, shift_vec)
    
    # return vectorized stencil
    return reshape(M_new, (nx*ny*nz,1) )
    
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