module SpinSimParams

    using StaticArrays
    using StatsBase # for sample()
    using Random, Distributions # for discrete sampling

## MAKE DICTIONARY OF INDEPENDENT VARIABLES
    function make_params(α, ω, n, bw = 0.5, γ = 2*pi*1e6, τ = 100e-6, dt = 2, ψ_0 = @SArray[1 0], ν0 = 10)

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
        f["n"] = n;
        f["integrated_sampling"] = false;
        f["tiles"] = false;
        
        # initial condition density matrix
        dim = size(ψ_0, 2);
        ρ_temp = [ψ_0[i]*ψ_0[j] for i = 1:dim, j = 1:dim];
        ρ0 = @SMatrix [ρ_temp[1,1] ρ_temp[1,2]; ρ_temp[2,1] ρ_temp[2,2]];
        f["ρ0"] = ρ0;
   
        # lattice parameters
        f["hlk"] = [0; 0; 0];
        f["θ"] = [0; 0; 0];
        f["k"] = [0; 0; 0];
        f["r"] = zeros(prod(n), 3);
    
        # correlation length
        f["local_M_on"] = false;

        return f

    end

## MAKE TEMPORARY params DICTIONARY FOR A GIVEN ITERATION AND CALCULATE DEPENDENT VARIABLES
    function make_temp_params(params, vars, I)
    
        # copy the parameter file
        f = copy(params);
        
        # reassign loop variables based on iteration
        for v in vars
            if typeof(v) == String
                idx = findall(y -> y == v, vars)[1];
                temp = f[v];
                f[v] = temp[I[idx]];
            else
                idx = findall(y -> y == v, vars)[1];
                for x in v
                    temp = f[x];
                    f[x] = temp[I[idx]];
                end
            end
        end

        ## calculate dependent variables ##

        # number of points in time
        f["nτ"] = convert(Int64, round(f["τ"]*f["γ"]/f["dt"]));
    
        # number of frequencies
        f["nfreq"] = prod(f["n"]);

        # ensemble variables
        if f["integrated_sampling"]
            
            ## INTEGRATED SAMPLING
            ν = collect(LinRange(f["ν0"] - f["bw"]/2, f["ν0"] + f["bw"]/2, f["nfreq"]));
            P = lorentzian.(ν, f["ν0"], 0.05)/sum(lorentzian.(ν, f["ν0"], 0.05));
            ρ = [f["ρ0"] for i = 1:f["nfreq"]];
            
            f["ν"] = ν;
            f["P"] = P;
            f["ρ_init"] = ρ;
            
            ## TILING, IF NEEDED
            if f["tiles"]
            
                # make the tiles
                f["r"] = make_tiles(f["r"], f["r_tiles"], f["tile_norm"])

                # do the tiling for the ensemble
                temp_ν = ν;
                temp_P = P;
                temp_ρ = ρ;
                for idx = 2:f["nfreq"]
                    temp_ν = cat(dims = 1, temp_ν, ν);
                    temp_P = cat(dims = 1, temp_P, P);
                    temp_ρ = cat(dims = 1, temp_ρ, ρ);
                end
                f["ν"] = temp_ν;
                f["P"] = temp_P/f["nfreq"];
                f["ρ_init"] = temp_ρ;
                f["nfreq"] = f["nfreq"]*size(r_tiles,1);
            
            end
        
        else
        
            ## DISCRETE SAMPLING
            x = collect(LinRange(f["ν0"] - f["bw"]/2, f["ν0"] + f["bw"]/2, f["nfreq"]));
            f["ν"] = sample(x, Weights(lorentzian.(x, f["ν0"], 0.05)), f["nfreq"]);
            f["P"] = ones(length(f["ν"]),1)./length(f["ν"]);
            f["ρ_init"] = [f["ρ0"] for i = 1:f["nfreq"]];
        
        end

        return f

    end

## MAKE A LATTICE
    function make_lattice(hlk, θ, n)
    
        npts = prod(n);

        # calculate the dimension of the lattice
        dim = size(hlk, 1);

        # zero fill the θ and hlk arrays and fill out n array with 1s
        while size(hlk, 1) < 3
            push!(hlk, 0);
        end
        while size(θ, 1) < 3
            push!(θ, 0);
        end

        # fill with unique integers
        f = sample(1:npts, n, replace = false)

        # replace integers with indices
        I = [findall(x -> x == temp, f)[1] for temp in f]

        # calculate positions
        r = zeros(npts, dim)
        i = 1;
        for idx in I
            x = (idx[1]-1)*hlk[1] + (idx[2]-1)*hlk[2]*cos(θ[1]) + (idx[3]-1)*hlk[3]*cos(θ[2])*sin(θ[3]);
            y = (idx[2]-1)*hlk[2]*sin(θ[1]) + (idx[3]-1)*hlk[3]*sin(θ[2])*sin(θ[3]);
            z = (idx[3]-1)*hlk[3]*cos(θ[3]);
            temp = [x; y; z];
            r[i,:] = round.(temp[1:dim], digits = 5);
            i += 1;
        end
        idx_r = sample(1:npts, npts, replace = false);

        return r, idx_r
    
    end

## DIVIDE LATTICE INTO TILES
    function make_tiles(r, r_tiles, tile_norm)
    
        # do the tiling for the positions
        R = ();
        nfreq = size(r,1);

        for i = 1:size(r_tiles,1)
            rt = zeros(nfreq, size(r,2));
            for j = 1:nfreq
                rt[j,:] = r_tiles[i,:];
            end
            idxt = sample(1:nfreq, nfreq, replace = false);
            tile = (r[idxt,:] .+ tile_norm*rt)/tile_norm;
            R = (R..., tile);
        end

        bigR = R[1]
        for i = 2:nfreq
            bigR = cat(dims = 1, bigR, R[i]);
        end
    
        return bigR
    
    end


## MAKE INDICES FOR GENERALIZED for LOOP
    function make_idx(vars, params)

        # create the tuple with the dimensions needed
        d = ();
        for v in vars
            # check for paired variables
            if typeof(v) == String
                f = params[v]
                lf = size(f,1)
                d = (d..., lf)
            else
                x = v[1];
                f = params[x]
                lf = size(f,1)
                d = (d..., lf)
            end
        end               

        # create array where element = index of element
        f = sample(1:prod(d), d, replace = false)
        I = [findall(x -> x == temp, f)[1] for temp in f]
        
        return I, d

    end

## CUSTOM DISTRIBUTIONS

    function lorentzian(x, μ, Γ)
        L = (1/π)*(Γ/2)/((x-μ)^2+(Γ/2)^2)
        return L
    end

    function gaussian(x,μ,σ)
        G = (1/(σ*sqrt(2*pi)))*exp(-(1/2)*((x.-μ)/σ).^2)
        return G
    end

    export make_params, make_temp_params, make_lattice, make_idx, make_tiles
    export lorentzian, gaussian

end