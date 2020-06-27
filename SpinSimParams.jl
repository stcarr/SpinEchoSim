module SpinSimParams

    using StaticArrays
    using StatsBase # for sample()
    using Random, Distributions # for discrete sampling


    mutable struct make_parameters

        # operators
        M_op
        Iz
        UL90
        UR90

        # time variables
        γ
        τ
        dt

        # interaction variables
        α
        ω

        # spin ensemble variables
        ν0
        bw
        dfreq
        ρ0
        int_sample::Bool # integrated vs discrete sampling
        nS::Int64
        r # positions of spins
        M_stencil # stencil for evalution of local M

        # dissipation parameters
        Lj # collection of jump operators

        function make_parameters(γ, τ, dt, α, ω, ν0, bw, dfreq, ρ0)

            # magnetization
            Ix = @SMatrix [0 1/2; 1/2 0];
            Iy = @SMatrix [0 -1im/2; 1im/2 0];
            Iz = @SMatrix [1/2 0; 0 -1/2];
            M_op = Ix + 1im*Iy;

            # pulse operators
            UL90 = exp(-1im*pi*Ix/2);
            UR90 = exp(1im*pi*Ix/2);

            new(M_op, Iz, UL90, UR90, γ, τ, dt, α, ω, ν0, bw, dfreq, ρ0, true, 1, nothing, nothing, nothing)

        end

    end

    mutable struct temp_parameters

        # operators
        M_op
        Iz
        UL90
        UR90

        # time variables
        γ
        τ
        dt
        nτ

        # interaction variables
        α
        ω

        # spin ensemble variables
        ν0
        bw
        dfreq
        ν
        P
        int_sample::Bool
        nS::Int64 # number of spins, if int_sample = false
        r # positions of spins
        M_stencil
        ρ_init
    
        # dissipation parameters
        Lj # collection of jump operators

        function temp_parameters(params, vars, I)

            # get the names of parameters
            fields = string.(fieldnames(make_parameters));

            # assign all variable values
            for x in fields
                f = getproperty(params, Symbol(x));
                @eval $(Symbol(x)) = $f;
            end

            # reassign loop variables based on this iteration
            for x in intersect(vars, fields)
                f = getproperty(params, Symbol(x));
                idx = findall(y -> y == x, vars)[1];
                f = f[I[idx]];
                @eval $(Symbol(x)) = $f;
            end

            # calculate temporary variables
            nτ = convert(Int64, round(τ*γ/dt))
        
            if (int_sample == true)
                ν = collect((ν0 - bw/2):dfreq:(ν0 + bw/2))
                P = lorentzian.(ν, ν0, 0.05)
            else
                ν = zeros(nS)
                d = Cauchy(ν0,0.05*0.5)            
                for v = 1:nS
                    v_here = rand(d)
                    while (abs(v_here - ν0) > bw/2)
                        v_here = rand(d)
                    end
                    ν[v] = v_here; # random samples
                end
                P = ones(nS)./nS; # uniform probability
            end
        
            P = P/sum(P) # normalize!
            ρ_init = [params.ρ0 for i = 1:size(ν,1)]

            new(M_op, Iz, UL90, UR90, γ, τ, dt, nτ, α, ω, ν0, bw, dfreq, ν, P, int_sample, nS, r, M_stencil, ρ_init, Lj)

        end

    end

    function make_idx(vars, params)

        fields = string.(fieldnames(make_parameters));

        # create the tuple with the dimensions needed
        d = ();
        p = 1;
        for x in intersect(vars, fields)
            f = getproperty(params, Symbol(x));
            lf = size(f,1)
            p = p*lf;
            d = (d..., lf);
        end
        f = sample(1:p, d, replace = false)

        # fill it with indices
        I = [findall(x -> x == temp, f)[1] for temp in f]

        return I, d
    end

    function lorentzian(x, μ, Γ)
        L = (1/π)*(Γ/2)/((x-μ)^2+(Γ/2)^2)
        return L
    end

    function gaussian(x,μ,σ)
        G = (1/(σ*sqrt(2*pi)))*exp(-(1/2)*((x.-μ)/σ).^2)
        return G
    end

    export make_parameters, temp_parameters, make_idx
    export lorentzian, gaussian

end