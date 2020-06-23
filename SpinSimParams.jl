module SpinSimParams

    using StaticArrays
    using StatsBase

    function make_params(γ, τ, dt, α, ω, ν0, bw, nfreq, ρ0)

        f = Dict();

        # magnetization
        f["Ix"] = @SMatrix [0 1/2; 1/2 0];
        f["Iy"] = @SMatrix [0 -1im/2; 1im/2 0];
        f["Iz"] = @SMatrix [1/2 0; 0 -1/2];
        f["M_op"] = f["Ix"] + 1im*f["Iy"];

        # pulse operators
        f["UL90"] = exp(-1im*pi*f["Ix"]/2);
        f["UR90"] = exp(1im*pi*f["Ix"]/2);

        # time variables
        f["γ"] = γ;
        f["τ"] = τ;
        f["dt"] = dt;

        # interaction variables
        f["α"] = α;
        f["ω"] = ω;

        # spin ensemble variables
        f["ν0"] = ν0;
        f["bw"] = bw;
        f["nfreq"] = nfreq;
        f["ρ0"] = ρ0;

        return f

    end

    function make_temp_params(params, vars, I)
    
        # copy the parameter file
        f = copy(params);

        # reassign loop variables based on iteration
        for v in vars
            idx = findall(y -> y == v, vars)[1];
            temp = f[v];
            f[v] = temp[I[idx]];
        end

        ## calculate temporary variables ##

        # number of points in time
        f["nτ"] = convert(Int64, round(f["τ"]*f["γ"]/f["dt"]));

        # frequency sample & weights
        f["ν"] = collect(LinRange(f["ν0"] - f["bw"]/2, f["ν0"] + f["bw"]/2, f["nfreq"]));
        f["P"] = lorentzian.(f["ν"], f["ν0"], 0.05)/sum(lorentzian.(f["ν"], f["ν0"], 0.05));

        # initial positions
        f["ρ_init"] = [f["ρ0"] for i = 1:f["nfreq"]];

        return f

    end

    function make_lattice(params)
    
        # load parameters
        hlk = params["hlk"];
        θ = params["θ"];
        nfreq = params["nfreq"];
    
        # calculate the dimension of the lattice
        dim = size(hlk, 1);

        # zero fill the θ and hlk arrays
        while size(hlk, 1) < 3
            push!(hlk, 0);
        end
        while size(θ, 1) < 3
            push!(θ, 0);
        end

        # create a tuple with the dimensions needed
        d = ();
        n = convert(Int64, round(nfreq^(1/dim)));
        for i = 1:dim
            d = (d..., n);
        end

        # fill with ones for the unused dimensions
        while size(d,1) < 3
            d = (d..., 1);
        end

        # fill with unique integers
        f = sample(1:nfreq, d, replace = false)

        # replace integers with indices
        I = [findall(x -> x == temp, f)[1] for temp in f]

        # calculate positions
        r = zeros(nfreq, dim)
        i = 1;
        for idx in I
            x = (idx[1]-1)*hlk[1] + (idx[2]-1)*hlk[2]*cos(θ[1]) + (idx[3]-1)*hlk[3]*cos(θ[2])*sin(θ[3]);
            y = (idx[2]-1)*hlk[2]*sin(θ[1]) + (idx[3]-1)*hlk[3]*sin(θ[2])*sin(θ[3]);
            z = (idx[3]-1)*hlk[3]*cos(θ[3]);
            temp = [x; y; z];
            r[i,:] = round.(temp[1:dim], digits = 5);
            i += 1;
        end
        idx_r = sample(1:nfreq, nfreq, replace = false);
    
        return r, idx_r
    
    end

    function make_idx(vars, params)

        # create the tuple with the dimensions needed
        d = ();
        p = 1;
        for v in vars
            f = params[v];
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

    export make_params, make_temp_params, make_idx, make_lattice
    export lorentzian, gaussian

end