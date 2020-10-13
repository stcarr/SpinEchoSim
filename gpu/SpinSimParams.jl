## CURRENTLY ONLY SET UP FOR PERIODIC BOUNDARY CONDITIONS

module SpinSimParams

    using StaticArrays
    using StatsBase
    using Random, Distributions
    using LinearAlgebra
    using CUDA

    ## MAKE DICTIONARY OF INDEPENDENT VARIABLES
    function make_params(α, n, bw = 0.5, γ = 2*pi*1e6, τ = 50e-6, dt = 2, ψ_0 = @SArray[1; 0], ν0 = 10)

        f = Dict();

        # magnetization
        f["Ix"] = convert.(Complex{Float32}, @SArray [0 1/2; 1/2 0]);
        f["Iy"] = convert.(Complex{Float32}, @SArray [0 -1im/2; 1im/2 0]);
        f["Iz"] = convert.(Complex{Float32}, @SArray [1/2 0; 0 -1/2]);
        f["Ip"] = f["Ix"] + 1im*f["Iy"];
        f["Im"] = f["Ix"] - 1im*f["Iy"];
        f["M_op"] = f["Ip"];

        # pulse operators
        f["U90"] = convert.(Complex{Float32}, exp(-1im*pi*f["Ix"]/2));
        f["U180"] = convert.(Complex{Float32}, exp(-1im*pi*f["Ix"]));

        # time variables
        f["γ"] = convert.(Float32, γ);
        f["τ"] = convert.(Float32, τ);
        f["dt"] = convert.(Float32, dt);

        # interaction variables
        f["α"] = convert.(Float32, α);

        # spin ensemble variables
        f["ν0"] = convert.(Float32, ν0);
        f["bw"] = convert.(Float32, bw);
        f["n"] = n;
        f["integrated_sampling"] = false;
        f["tiles"] = false;

        # initial condition
        f["ψ0"] = convert.(Complex{Float32}, ψ_0);

        # correlation length, off by default
        f["local_M_on"] = true;

        # dissipation, off by default
        f["dissipation"] = true;
    
        # cpmg variables
        f["echo_time"] = 10e-6;
        f["num_echoes"] = 50;

        return f

    end

    ## MAKE TEMPORARY params DICTIONARY FOR A GIVEN ITERATION AND CALCULATE DEPENDENT VARIABLES
    function make_temp_params(params, vars, I)

        # copy the parameter file
        f = copy(params);

        # reassign loop variables based on iteration
        num_idx_assign = 0;
        tupI = ()
        for idx_l = 1:length(I)
            tupI = (tupI..., I[idx_l])
        end

        for v in vars
            if typeof(v) == String
                idx = findall(y -> y == v, vars)[1];
                temp = f[v];
                f[v] = temp[I[idx]];
                num_idx_assign += 1 
            else
                L = length.(v)
                idx = findall(y -> y == v, vars)[1];
                if maximum(L)[1] > 1 && [typeof(x) == String for x in v] != [true for x in v]
                    idx_h = tupI[(num_idx_assign+1):(num_idx_assign+maximum(L)[1])]
                    temp = f[v[1]]
                    f[v[1]] = temp[idx_h...]
                    x = v[2]
                    for idx_x = 1:length(x)
                        temp = f[x[idx_x]]
                        f[x[idx_x]] = temp[idx_h[idx_x]]
                    end
                    num_idx_assign += maximum(L)[1]
                else    
                    for x in v
                        temp = f[x];
                        f[x] = temp[I[idx]];
                    end
                    num_idx_assign += 1
                end
            end
        end

        ## calculate dependent variables ##

        # set some dependent defaults (for spatial variation, there's probably a general way to do this)
        n = f["n"]
        if ~haskey(f, "r")
            if length(n) == 1
                f["k"] = [0];
                f["r"] = fill(0, n)
                f["spin_idx"] = [(i) for i = 1:n[1]]
            elseif length(n) == 2
                f["k"] = [0; 0];
                f["r"] = fill([0; 0], n)
                f["spin_idx"] = [(i, j) for i = 1:n[1], j = 1:n[2]]
            else
                f["k"] = [0; 0; 0];
                f["r"] = fill([0; 0; 0], n)
                f["spin_idx"] = [(i, j, k) for i = 1:n[1], j = 1:n[2], k = 1:n[3]]
            end
        end

        # number of points in time
        f["nτ"] = convert(Int64, round(f["τ"]*f["γ"]/f["dt"]));

        # number of frequencies
        f["nfreq"] = prod(f["n"]);

        ## INTEGRATED SAMPLING
        if f["integrated_sampling"]

            ## TILING, IF NEEDED
            if f["tiles"]

                # create the tile distributions
                tile_ν = collect(LinRange(f["ν0"] - f["bw"]/2, f["ν0"] + f["bw"]/2, prod(f["tile_dims"])))
                tile_P = lorentzian.(tile_ν, f["ν0"], 0.05)/sum(lorentzian.(tile_ν, f["ν0"], 0.05))

                # make the tiles
                big_ν, big_P = make_freq_tiles(tile_ν, tile_P, f["n"], f["tile_dims"])

                f["ν"] = big_ν
                f["P"] = big_P
                f["ψ_init"] = fill(f["ψ0"], f["n"])

            else

                ν = collect(LinRange(f["ν0"] - f["bw"]/2, f["ν0"] + f["bw"]/2, f["nfreq"]))
                P = lorentzian.(ν, f["ν0"], 0.05)/sum(lorentzian.(ν, f["ν0"], 0.05))

                f["ν"] = reshape(ν, f["n"])
                f["P"] = reshape(P, f["n"])
                f["ψ_init"] = fill(f["ψ0"], f["n"]);

            end

        else

            ## DISCRETE SAMPLING
            x = collect(LinRange(f["ν0"] - f["bw"]/2, f["ν0"] + f["bw"]/2, f["nfreq"]))
            f["ν"] = convert.(Complex{Float32}, sample(x, Weights(lorentzian.(x, f["ν0"], 0.05)), f["n"]))
            f["P"] = convert.(Complex{Float32}, fill(1/prod(f["n"]), f["n"]))
            f["ψ_init"] = fill(f["ψ0"], f["n"])

        end

        # calculate dissipation jump operators, if needed
        if f["dissipation"]

            Γ = f["Γ"]

            Ljs = [];
            push!(Ljs, sqrt(Γ[1])*f["Ip"])
            push!(Ljs, sqrt(Γ[2])*f["Im"])
            push!(Ljs, sqrt(Γ[3])*f["Iz"])

            f["Lj"] = Ljs;

        end

        return f

    end

    ## MAKE A LATTICE
    function make_lattice(hlk, θ, n, pbc)

        npts = prod(n)

        # calculate the dimension of the lattice
        dim = size(hlk, 1)

        # zero fill the θ and hlk arrays and fill out n array with 1s
        hlk_h = copy(hlk)
        θ_h = copy(θ)
        n_h = tuple(copy(collect(n))...) # i hate tuples

        while size(hlk_h, 1) < 3
            push!(hlk_h, 0)
            n_h = (n_h..., 1)
        end
        while size(θ_h, 1) < 3
            push!(θ_h, 0)
        end

        # calculate coordinates
        f = sample(1:npts, n, replace = false)
        idx_list = [findall(x -> x == temp, f)[1] for temp in f]

        # array to save positions
        r = Array{Any}(undef, n)
        spin_idx = Array{Any}(undef, n)

        # calculate positions
        for idx in idx_list

            # have to retrieve individual values from idx (bc of stupid cartesianindex types)
            n_idx = [];
            for idx_h = 1:length(idx)
                push!(n_idx, idx[idx_h])
            end

            # fill it out with ones for the unit dimensions
            while length(n_idx) < 3
                push!(n_idx, 1)
            end

            # assign local indices based on periodic or non periodic bc
            if pbc
                x_idx = mod(n_idx[1] - 1 + n_h[1]/2, n_h[1]) - n_h[1]/2
                y_idx = mod(n_idx[2] - 1 + n_h[2]/2, n_h[2]) - n_h[2]/2
                z_idx = mod(n_idx[3] - 1 + n_h[3]/2, n_h[3]) - n_h[3]/2
            else
                x_idx = n_idx[1] - 1
                y_idx = n_idx[2] - 1
                z_idx = n_idx[3] - 1
            end

            # calculate the positions
            x = x_idx*hlk_h[1] + y_idx*hlk_h[2]*cos(θ_h[1]) + z_idx*hlk_h[3]*cos(θ_h[2])*sin(θ_h[3])
            y = y_idx*hlk_h[2]*sin(θ_h[1]) + z_idx*hlk_h[3]*sin(θ_h[2])*sin(θ_h[3])
            z = z_idx*hlk_h[3]*cos(θ_h[3])

            # assign only the non unit dimensions
            temp_r = [x, y, z]
            temp_idx = (n_idx[1], n_idx[2], n_idx[3])
            r[idx] = temp_r[1:dim]
            spin_idx[idx] = temp_idx[1:dim]

        end

        return r, spin_idx

    end

    ## DIVIDE LATTICE INTO TILES
    function make_freq_tiles(ν, P, n, tile_dims)

        # to hold the frequencies
        big_ν = Array{Any}(undef, n)
        big_P = Array{Any}(undef, n)

        # size of the tiles in each direction
        tile_dims = (10, 10)

        # calculate the number of times in each direction
        n_h = copy(collect(n))
        num_tiles = tuple(convert.(Int64, n_h./tile_dims)...)

        # create some coordinates
        f = sample(1:prod(num_tiles), num_tiles, replace = false)
        idx_list = [findall(y -> y == x, f)[1] for x in f]

        for idx in idx_list

            # convert to non cartesian index
            tile_coord = []
            for ii = 1:length(idx)
                push!(tile_coord, idx[ii])
            end

            n0 = (tile_coord .- 1).*tile_dims .+ 1
            nf = tile_coord.*tile_dims

            # get subspace of frequency mesh from n0, nf-- couldn't think of a better way to do it for varying length(n), alas
            if length(n) == 1
                temp_idx = collect(convert.(Int64, round.(LinRange(n0[1], nf[1], tile_dims[1]))))
            elseif length(n) == 2
                temp_idx_x = collect(convert.(Int64, round.(LinRange(n0[1], nf[1], tile_dims[1]))))
                temp_idx_y = collect(convert.(Int64, round.(LinRange(n0[2], nf[2], tile_dims[2]))))
                temp_idx = [(temp_idx_x[i], temp_idx_y[j]) for i = 1:tile_dims[1], j = 1:tile_dims[2]]
            else
                temp_idx_x = collect(convert.(Int64, round.(LinRange(n0[1], nf[1], tile_dims[1]))))
                temp_idx_y = collect(convert.(Int64, round.(LinRange(n0[2], nf[2], tile_dims[2]))))
                temp_idx_z = collect(convert.(Int64, round.(LinRange(n0[3], nf[3], tile_dims[3]))))
                temp_idx = [(temp_idx_x[i], temp_idx_y[j], temp_idx_z[k]) for i = 1:tile_dims[1], j = 1:tile_dims[2], k = 1:tile_dims[3]]
            end

            # fill with spin sampling
            shuff = sample(1:prod(tile_dims), length(ν), replace = false)
            temp_ν = ν[shuff]
            temp_P = P[shuff]
            idx3 = 1
            for idx2 in temp_idx
                big_ν[idx2...] = temp_ν[idx3]
                big_P[idx2...] = temp_P[idx3]
                idx3 += 1
            end

        end

        # normalize big_P
        big_P = big_P./prod(num_tiles)

        return big_ν, big_P

    end

    ## MAKE INDICES FOR GENERALIZED for LOOP
    function make_idx(vars, params)

        # create the tuple with the dimensions needed
        d = ();
    
        if isempty(vars)
        
            d = (1)
            f = [1]
            I = [findall(x -> x == temp, f)[1] for temp in f]
        
        else
    
            for v in vars

                # check for paired variables
                if typeof(v) == String
                    f = params[v]
                    lf = size(f,1)
                    d = (d..., lf)
                else
                    L = length.(v)
                    # check variables that depend on more than one other variable
                    if maximum(L)[1] > 1 && [typeof(x) == String for x in v] != [true for x in v]
                        x = v[1]
                        f = params[x]
                        lf = size(f)
                        for lx in lf
                            d = (d..., lx)
                        end
                    else
                        x = v[1];
                        f = params[x]
                        lf = size(f,1)
                        d = (d..., lf)
                    end
                end
            end               

            # create array where element = index of element
            f = sample(1:prod(d), d, replace = false)
            I = [findall(x -> x == temp, f)[1] for temp in f]
        
        end

        return I, d

    end

    ## MAKE STENCIL
    function make_stencil(hlk, θ, n, r, ξ, pbc, p)

        if pbc

            # calculate the stencil
            stencil = zeros(Float64, size(r))
            for temp_r in r
                idx = findall(y -> y == temp_r, r)[1]
                stencil[idx] = 1/((norm(temp_r)/ξ)^p + 1)
            end
        
            # set self coupling to zero
            stencil[1,1] = 0;

        else

            # making 4 tiles of the lattice
            new_n = tuple(collect(2 .*n)...)

            # make a lattice 4x the size
            r_h, spin_idx_h = make_lattice(hlk, θ, new_n, true)

            # calculate the stencil
            stencil = zeros(Float64, size(r_h))
            for temp_r in r_h
                idx = findall(y -> y == temp_r, r_h)[1]
                stencil[idx] = 1/(norm(temp_r)/ξ + 1)
            end

        end

        return stencil

    end

    function make_sparse_stencil(r, ξ, n, thresh1, thresh2, decay_pow, decay_coeff)

        p = 4; # 1/(r/ξ)^p+1
        thresh = thresh1;
        n_thresh = thresh2;

        # calculate the stencil with zeros
        stencil = zeros(Float64, size(r))
        for temp_r in r
            idx = findall(y -> y == temp_r, r)[1]
            temp = 1/((norm(temp_r)/ξ)^p + 1)
            if temp < thresh
                temp = 0
            end
            stencil[idx] = temp;
        end

        # add the smoothness
        idx_list = findall(y -> y == 0, stencil)

        for idx in idx_list

            # initial values
            x0 = idx[1];
            y0 = idx[2];
            r0 = r[idx];
            d0 = 10000;
            nearest_idx = (0,0)

            # continue until find a non-zero value
            for x_shift = 1:n[1]

                # move by 1 row, modular
                x = mod(x0 + x_shift - 1, n[1]) + 1

                # scan columns
                for y_shift = 1:n[2]

                    # move by 1 column
                    y = mod(y0 + y_shift - 1, n[2]) + 1

                    # calculate the distance
                    d = norm(r0 - r[x,y])

                    # calculate the stencil value
                    temp = stencil[x,y]

                    # save if it's closer & non-zero
                    if temp > 0
                        if d < d0
                            d0 = d
                            nearest_idx = (x, y)
                        end
                    end
                end
            end

            temp = exp(-(decay_coeff*d0)^decay_pow)*stencil[nearest_idx...]
            if temp > n_thresh
                stencil[idx] = temp
            end

        end

        # no self coupling
        stencil[1,1] = 0;

        # stencil = sparse(stencil)
        return stencil

    end

    ## CHECK FOR PERIODIC BOUNDARY CONDITIONS AND DO THE SHIFT
    function shift_stencil_pbc(stencil, P, spin_idx, n)

        stencil = CuArray(stencil)
        bigS = CUDA.zeros(n[1], n[2], n[1], n[2])

        for v in spin_idx
            shift = (spin_idx[v...][1] - 1, spin_idx[v...][2] - 1)
            temp = P.*circshift(stencil, shift)
            bigS[:,:,v[1],v[2]] = temp/sum(temp)
        end

        return bigS

    end   

    ## OLD ## CHECK FOR PERIODIC BOUNDARY CONDITIONS AND DO THE SHIFT
    function shift_stencil(stencil, P, spin_idx, n, pbc)

        # shift based on the current spin  
        stencil_h = circshift(copy(stencil), spin_idx)

        # if using the big_stencil approach, take only the correct subspace of the big stencil
        if ~pbc

            # conditionals cuz idk how else to do it
            if length(n) == 1 
                return stencil_h[1:n[1]] 
            elseif length(n) == 2
                return stencil_h[1:n[1],1:n[2]] 
            else
                return stencil_h[1:n[1],1:n[2],1:n[3]]
            end

        # otherwise just spit it back
        else

            return stencil_h

        end

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

    export lorentzian, gaussian
    export make_params, make_temp_params, make_lattice, make_freq_tiles, make_idx, make_stencil, make_sparse_stencil, shift_stencil_pbc

end


