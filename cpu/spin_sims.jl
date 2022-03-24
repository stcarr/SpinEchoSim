## ONLY WORKS FOR BOTH DISSIPATION AND LOCAL M

include("../lib/liouville_tools.jl")

using .LiouvilleTools
using LinearAlgebra

function spin_echo_sim_liouville(params)

    # initialize M_list
    M_list = [];
    Mz_list = [];

    pl1 = params["pulse_length1"]*params["γ"];
    pl2 = params["pulse_length2"]*params["γ"];

    UL90 = params["U90"];
    UR90 = UL90'

    UL180 = params["U180"];
    UR180 = UL180'

    null_operator = 0*UL90

    ρ_list_L = [dm_H2L(ψ*ψ') for ψ in params["ψ_init"]];
    t0 = 0.0;

    # 90 pulse
    if (pl1 == 0)
        t1 = t0;
        ρ_list_L = [dm_H2L(UL90* dm_L2H(ρ_L) *UR90) for ρ_L in ρ_list_L];
    else
        nt_p1 = round(pl1/params["dt"])
        p90 = (params["R90"]/(nt_p1))
        ρ_list_L, M_list, Mz_list, t1 = time_propagate_liouville(ρ_list_L, M_list, Mz_list, t0, params["dt"], nt_p1, p90, params)
    end

    # first tau
    # center 0 at center of pulse 1, tau at center of pulse 2
    tau = params["dt"]*params["nτ"]
    eff_tau =  tau - (pl1 + pl2)/2.0;
    eff_nt = round(eff_tau/params["dt"])
    ρ_list_L, M_list, Mz_list, t2 = time_propagate_liouville(ρ_list_L, M_list, Mz_list, t1, params["dt"], eff_nt, null_operator, params)

    # 180 pulse
    if (pl2 == 0)
        t3 = t2;
        ρ_list_L = [dm_H2L(UL180* dm_L2H(ρ_L) *UR180) for ρ_L in ρ_list_L];
    else
        nt_p2 = round(pl2/params["dt"])
        p180 = (params["R180"]/(nt_p2))
        ρ_list_L, M_list, Mz_list, t3 = time_propagate_liouville(ρ_list_L, M_list, Mz_list, t2, params["dt"], nt_p2, p180, params)
    end

    # end of 180 pulse to 3 tau
    eff_tau =  2.0*tau - pl2/2.0;
    eff_nt = round(eff_tau/params["dt"])
    ρ_list_L, M_list, Mz_list, t4 = time_propagate_liouville(ρ_list_L, M_list, Mz_list, t3, params["dt"], eff_nt, null_operator, params)

    t_max = t4/params["γ"]*1e6
    t_origin = t1/(2.0*params["γ"])*1e6;

    t_arr = collect(  range(-t_origin, stop=t_max-t_origin, length=size(M_list,1))  )

  return M_list, Mz_list, t_arr

end

function time_propagate_liouville(ρ_list_L, M_list, Mz_list, t0, dt, nsteps, M_c, params)

    # spectrum info
    ν0 = params["ν0"] # central freq.
    ν = params["ν"] # spin freqs.
    P = params["P"] # spin weights
    nS = params["nfreq"] # number of spins

    # operators
    M_op = params["M_op"]
    Ix = params["Ix"]
    Iy = params["Iy"]
    Iz = params["Iz"]
    #M_c, given, constant operator

    # additional values
    n = params["n"]
    spin_idx = params["spin_idx"]

    # interaction parameters
    α_mat = params["α_mat"]
    #α_y = params["α_y"]
    #α_z = params["α_z"]

    # initial time
    t = t0;

    # jump operators
    Lj_list = params["Lj"]
    J_L = JumpsToSuper(Lj_list) # get dissipative super operator (assumed constant in time)

    # initial magnetization
    M_L = leftop_H2L(M_op)
    Ix_L = leftop_H2L(Ix)
    Iy_L = leftop_H2L(Iy)
    Iz_L = leftop_H2L(Iz)

    M_eval = [tr_L(M_L*ρ) for ρ in ρ_list_L]
    Mx = [tr_L(Ix_L*ρ) for ρ in ρ_list_L]
    My = [tr_L(Iy_L*ρ) for ρ in ρ_list_L]
    Mz = [tr_L(Iz_L*ρ) for ρ in ρ_list_L]

    M = sum(P.*M_eval);

    # prepare the stencils
    M_stencil = params["M_stencil"]
    M_stencil_vec = shift_stencil(M_stencil, P, spin_idx, n)

    # calculate local M
    #M_local = [sum(M_eval.*M_stencil_vec[:,:,v[1],v[2]]) for v in spin_idx]
    Mx_local = [sum(Mx.*M_stencil_vec[:,:,v[1],v[2]]) for v in spin_idx]
    My_local = [sum(My.*M_stencil_vec[:,:,v[1],v[2]]) for v in spin_idx]
    Mz_local = [sum(Mz.*M_stencil_vec[:,:,v[1],v[2]]) for v in spin_idx]

    # time evolve
    for idx = 1:nsteps

        t += dt;

        # calculate interaction
        #int_x = α_x*map(x -> (1/4)*[0 conj(x); x 0] - (1/4)*[0 x*exp(-2im*ν0*t); conj(x)*exp(2im*ν0*t) 0], Mx_local)
        #int_y = α_x*map(x -> (1/4)*[0 conj(x); x 0] - (1/4)*[0 x*exp(-2im*ν0*t); conj(x)*exp(2im*ν0*t) 0], Mx_local)
        #int_x = Mx_local .* cos(ν0*t) + My_local .* sin(ν0*t)
        #int_y = My_local .* cos(ν0*t) - Mx_local .* sin(ν0*t)
        #int_z = α_z .* Mz_local

        int = 0 .* Mx_local .* [Ix];
        for d2 = 1:3

            if d2 == 1
                M_h = Mx_local .* cos(ν0*t) + My_local .* sin(ν0*t)
            elseif d2 == 2
                M_h = My_local .* cos(ν0*t) - Mx_local .* sin(ν0*t)
            elseif d2 == 3
                M_h = Mz_local
            end

            for d1 = 1:3

                if d1 == 1
                    I_h = Ix .* cos(ν0*t) + Iy .* sin(ν0*t)
                elseif d1 == 2
                    I_h = Iy .* cos(ν0*t) - Ix .* sin(ν0*t)
                elseif d1 == 3
                    I_h = Iz
                end

                int = int .+ α_mat[d1,d2] .* M_h .* [I_h];

            end
        end
        #int =  α.*(Mx_local.*[Ix] + My_local.*[Iy]) + α_z .* Mz_local.*[Iz]
        # calculate hamiltonian
        H_H = -(ν.-ν0).*[Iz] .- int .+ ones(size(ν)).*[M_c]
        H_L = [HamToSuper(H) for H in H_H]

        # calculate propagators, adding in dissipation
        U_L = [exp(( -1im*H + J_L )*dt) for H in H_L]

        # time evolve
        ρ_list_L = U_L.*ρ_list_L

        # update M and save value
        M_eval = [tr_L(M_L*ρ) for ρ in ρ_list_L]
        Mx = [tr_L(Ix_L*ρ) for ρ in ρ_list_L]
        My = [tr_L(Iy_L*ρ) for ρ in ρ_list_L]
        Mz = [tr_L(Iz_L*ρ) for ρ in ρ_list_L]

        M = sum(P.*M_eval);
        Mz_save = sum(P.*Mz);
        push!(M_list, M)
        push!(Mz_list, Mz_save)

        # calculate local M
        #M_local = [sum(M_eval.*M_stencil_vec[:,:,v[1],v[2]]) for v in spin_idx]
        Mx_local = [sum(Mx.*M_stencil_vec[:,:,v[1],v[2]]) for v in spin_idx]
        My_local = [sum(My.*M_stencil_vec[:,:,v[1],v[2]]) for v in spin_idx]
        Mz_local = [sum(Mz.*M_stencil_vec[:,:,v[1],v[2]]) for v in spin_idx]

    end

    return ρ_list_L, M_list, Mz_list, t

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

function shift_stencil(stencil, P, spin_idx, n)

    bigS = zeros(n[1], n[2], n[1], n[2])

    for v in spin_idx
        shift = (spin_idx[v...][1] - 1, spin_idx[v...][2] - 1)
        temp = P.*circshift(stencil, shift)
        bigS[:,:,v[1],v[2]] = temp # /sum(temp)
    end

    return bigS

end
