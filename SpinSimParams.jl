module SpinSimParams

    using StaticArrays
    using StatsBase

    function make_params(α, ω, nfreq, bw = 0.5, γ = 2*pi*1e6, τ = 100e-6, dt = 2, ψ_0 = @SArray[1 0], ν0 = 10)

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
        f["sampling_type"] = "integrated";
    
        dim = size(ψ_0, 2);
        ρ_temp = [ψ_0[i]*ψ_0[j] for i = 1:dim, j = 1:dim];
        ρ0 = @SMatrix [ρ_temp[1,1] ρ_temp[1,2]; ρ_temp[2,1] ρ_temp[2,2]];
        
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
        if f["sampling_type"] == "integrated"
            f["ν"] = collect(LinRange(f["ν0"] - f["bw"]/2, f["ν0"] + f["bw"]/2, f["nfreq"]));
            f["P"] = lorentzian.(f["ν"], f["ν0"], 0.05)/sum(lorentzian.(f["ν"], f["ν0"], 0.05));
        else
            x = collect(LinRange(f["ν0"] - f["bw"]/2, f["ν0"] + f["bw"]/2, f["nfreq"]));
            f["ν"] = sample(x, Weights(lorentzian.(x, f["ν0"], 0.05)), f["nfreq"]));
            f["P"] = ones(length(f["ν"]),1)./length(f["ν"]);
        end
            
        # initial positions
        f["ρ_init"] = [f["ρ0"] for i = 1:f["nfreq"]];

        return f

    end

    function make_tiles(params)
        
        temp_ρ = params["ρ_init"]
        temp_ν = params["ν"]
        temp_P = params["P"]
        for idx = 2:params["nfreq"]
            temp_ρ = cat(dims = 1, temp_ρ, params["ρ_init"]);
            temp_ν = cat(dims = 1, temp_ν, params["ν"]);
            temp_P = cat(dims = 1, temp_P, params["P"]);
        end

        params["ρ_init"] = temp_ρ;
        params["ν"] = temp_ν;
        params["P"] = temp_P/params["nfreq"];
        params["nfreq"] = params["nfreq"]*params["nfreq"];
    
    end

    function make_tiles_R(r, params, globalM = true)
        
        R = ();
        nfreq = params["nfreq"];
        
        for i = 1:nfreq
            rt = zeros(nfreq, 2);
            for j = 1:nfreq
                rt[j,:] = r[i,:];
            end
            idxt = sample(1:nfreq, nfreq, replace = false);
            tile = (r[idxt,:] .+ sqrt(nfreq)*rt)/sqrt(nfreq);
            R = (R..., tile);
        end

        if globalM
            bigR = R[1]
            for i = 2:nfreq
                bigR = cat(dims = 1, bigR, R[i]);
            end

            R = bigR;
        end
        
        return R
    
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