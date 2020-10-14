## DISCRETE, PERIODIC BC, LIOUVILLE, DISSIPATION, STENCIL ONLY ##

module SpinSimParams

    using StaticArrays
    using StatsBase
    using Random, Distributions
    using LinearAlgebra
    using CUDA

    ## MAKE DICTIONARY OF INDEPENDENT VARIABLES
    function make_params()

        f = Dict();

        # magnetization
        f["Ix"] = convert.(Complex{Float32}, @SArray [0 1/2; 1/2 0])
        f["Iy"] = convert.(Complex{Float32}, @SArray [0 -1im/2; 1im/2 0])
        f["Iz"] = convert.(Complex{Float32}, @SArray [1/2 0; 0 -1/2])
        f["Ip"] = f["Ix"] + 1im*f["Iy"]
        f["Im"] = f["Ix"] - 1im*f["Iy"]
        f["M_op"] = f["Ip"]

        # pulse operators
        f["flip_angle"] = π/2
        f["phases"] = (0, π/2)

        # time variables
        f["γ"] = convert.(Float32, 2*pi*1e6)
        f["τ"] = convert.(Float32, 50e-6)
        f["dt"] = convert.(Float32, 2)

        # interaction variables
        f["α"] = convert.(Float32, 0)

        # spin ensemble variables
        f["ν0"] = convert.(Float32, 10)
        f["bw"] = convert.(Float32, 0.5)
        f["n"] = (100, 100)
        f["line_width"] = 0.05

        # initial condition
        f["ψ0"] = convert.(Complex{Float32}, @SArray[1; 0])
    
        # cpmg variables
        f["echo_time"] = 10e-6
        f["num_echoes"] = 50

        return f

    end

    ## MAKE TEMPORARY params DICTIONARY FOR A GIVEN ITERATION AND CALCULATE DEPENDENT VARIABLES
    function make_temp_params(params, vars, I)

        # copy the parameter file
        f = copy(params);

        # how many of the variables you've assigned (counter)
        num_idx_assign = 0;
    
        # rewrite I as a tuple (from cartesian index)
        tupI = ()
        for idx_l = 1:length(I)
            tupI = (tupI..., I[idx_l])
        end

        # figuring out the grouping of variables and assigning them
        for v in vars
        
            # if it's a single variable, just assign it
            if typeof(v) == String
                idx = findall(y -> y == v, vars)[1]; # find which position it is in the variable list
                temp = f[v]; # load the dictionary value
                f[v] = temp[I[idx]]; # reassign based on iteration index
                num_idx_assign += 1 # progress counter
            
            # check for other options: 
            # option 1: tuple of strings (x, y): two variables x and y vary together
            # option 2: tuple of tuples and strings (z(x,y), (x, y)): variable z dependent on two other varibles x and y
            else
            
                # check the length of each element of v to find tuples (ie to check for option 2)
                L = length.(v)
                idx = findall(y -> y == v, vars)[1];
            
                # option 2
                if maximum(L)[1] > 1 && [typeof(x) == String for x in v] != [true for x in v]
                
                    # if it is a tuple of var and tuples, then it's a set of the form (z(x,y), (x,y))
                    # ie x and y vary separately, but z depends on the value of x and y and must match it
                    # ie the ijth iteration of z must occur with the ith iteration of x and jth iteration of y
                
                    idx_h = tupI[(num_idx_assign+1):(num_idx_assign+maximum(L)[1])] # load the range of indices needed (ie x and y indices)
                    temp = f[v[1]] # load z(x,y)
                    f[v[1]] = temp[idx_h...] # match z(x,y) to x and y
                    x = v[2] # load the tuple (x,y)
                    for idx_x = 1:length(x)
                        temp = f[x[idx_x]] # load x, y
                        f[x[idx_x]] = temp[idx_h[idx_x]] # assign x, y
                    end
                    num_idx_assign += maximum(L)[1] # progress counter by amount = number of independent variables (ie how many xs, ys)
                
                # option 1
                else    
                    for x in v
                        temp = f[x]; # load x, y
                        f[x] = temp[I[idx]]; # assign x, y using the same index
                    end
                    num_idx_assign += 1 # progress counter
                end
            end
        end

        ## calculate dependent variables ##
    
        # rotation operators
        phases = f["phases"]
        flip_angle = f["flip_angle"]
        R90 = f["Ix"]*cos(phases[1]) + f["Iy"]*sin(phases[1])
        R180 = f["Ix"]*cos(phases[2]) + f["Iy"]*sin(phases[2])
        f["U90"] = convert.(Complex{Float32}, exp(-1im*flip_angle*R90));
        f["U180"] = convert.(Complex{Float32}, exp(-1im*2*flip_angle*R180));

        # number of points in time
        f["nτ"] = convert(Int64, round(f["τ"]*f["γ"]/f["dt"]));

        # number of frequencies
        f["nfreq"] = prod(f["n"]);

        # discrete frequency sampling
        x = collect(LinRange(f["ν0"] - f["bw"]/2, f["ν0"] + f["bw"]/2, f["nfreq"]))
        f["ν"] = convert.(Complex{Float32}, sample(x, Weights(lorentzian.(x, f["ν0"], f["line_width"])), f["n"]))
        f["P"] = convert.(Complex{Float32}, fill(1/prod(f["n"]), f["n"]))
        f["ψ_init"] = fill(f["ψ0"], f["n"])

        # calculate dissipation jump operators
        Γ = f["Γ"]
        Ljs = [];
        push!(Ljs, sqrt(Γ[1])*f["Ip"])
        push!(Ljs, sqrt(Γ[2])*f["Im"])
        push!(Ljs, sqrt(Γ[3])*f["Iz"])
        f["Lj"] = Ljs;

        return f

    end

    ## MAKE A LATTICE
    function make_lattice(hlk, θ, n)

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

            # assign local indices for periodic bc
            x_idx = mod(n_idx[1] - 1 + n_h[1]/2, n_h[1]) - n_h[1]/2
            y_idx = mod(n_idx[2] - 1 + n_h[2]/2, n_h[2]) - n_h[2]/2
            z_idx = mod(n_idx[3] - 1 + n_h[3]/2, n_h[3]) - n_h[3]/2

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

    ## MAKE INDICES FOR GENERALIZED for LOOP
    function make_idx(vars, params)

        # create the tuple with the dimensions needed
        d = ();
    
        # for no variables, return 1
        if isempty(vars)
            d = (1)
            f = [1]
            I = [findall(x -> x == temp, f)[1] for temp in f]
        else
            
            # pick an element of the variable list
            for v in vars
            
                # if it is a single variable, count how many times it changes and add that to the dimension of the idx list
                if typeof(v) == String
                    f = params[v]
                    lf = size(f,1)
                    d = (d..., lf)
                else
                    
                    # otherwise, if it is a tuple, check if it's a tuple of single variables or a tuple of variables and tuples
                    L = length.(v)
                    if maximum(L)[1] > 1 && [typeof(x) == String for x in v] != [true for x in v]
                    
                        # if it is a tuple of var and tuples, then it's a set of the form (z(x,y), (x,y))
                        # ie x and y vary separately, but z depends on the value of x and y and must match it
                        # ie the ijth iteration of z must occur with the ith iteration of x and jth iteration of y
                        # z does not get its own dimension, therefore, so only assign the lengths of x and y as dimensions for the idx list
                        x = v[1]
                        f = params[x]
                        lf = size(f) # z is length(x) x length(y) so the dimensions are saved by checking z
                        for lx in lf
                            d = (d..., lx)
                        end
                    
                    else
                    
                        # finally, if v is a tuple of variables, they all vary together
                        # ie for (x,y) the ith iteration of x must occur with the ith iteration of y
                        # therefore assign a single dimension to all of them
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
    function make_stencil(r, ξ, p)

        # calculate the stencil
        stencil = zeros(Float64, size(r))
        for temp_r in r
            idx = findall(y -> y == temp_r, r)[1]
            stencil[idx] = 1/((norm(temp_r)/ξ)^p)
        end

        # set self coupling to zero
        stencil[1,1] = 0;

        return stencil

    end

    ## CUSTOM DISTRIBUTIONS
    function lorentzian(x, μ, Γ)
        L = (1/π)*(Γ/2)/((x-μ)^2+(Γ/2)^2)
        return L
    end

    function gaussian(x, μ, σ)
        G = (1/(σ*sqrt(2*pi)))*exp(-(1/2)*((x.-μ)/σ).^2)
        return G
    end

    export lorentzian, gaussian
    export make_params, make_temp_params, make_lattice, make_idx, make_stencil

end


