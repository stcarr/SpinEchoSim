module MTools

    include("SpinSimParams.jl")
    using .SpinSimParams
    using Plots

## CALCULATE THE DATA

    function getMstats(dat, idx_w_f=0.75, sym_w=50, min_cutoff=0.02, mean_range=50, rmse=false)

        # idx_w_f :  number of time indices on each side to keep as a fraction of ntau (to accomodate varying tau)
        # sym_w : number of time indices on each side to evaluate symmetry
        # min_cutoff : value of M to assess as a "dip"
        # mean_range : number of time indices on each side for evaluating the spread
        # rmse : alternate method to calculate asymmetry, by taking squared difference of signal when mirrored over ie (y(xL) - y(xR))^2

        # calculate the dimensions of M_list
        idx_list, d = make_idx(dat["vars"], dat)
        M_list = dat["M_list"];

        # fix the "unwrapping" of multi-dim arrays by JSON
        tempM_list = Array{Any}(undef, d)
        for i in idx_list
            tempI = ();
            tempM = M_list;
            for idx = 1:length(i)
                tempI = (tempI..., i[idx])
                tempM = tempM[i[idx]];
            end
            # tempI = reverse(tempI)
            tempM_list[tempI...] = tempM
        end
        M_list = tempM_list;

        M_stats = Dict();

        for i in idx_list

            # assign dependent/temporary variables
            temp_dat = make_temp_params(dat, dat["vars"], i);

            # load and reformat M, t, nτ
            M = [ x["re"] + im*x["im"] for x in M_list[i] ];
            τ = temp_dat["τ"];
            nτ = temp_dat["nτ"];
            t = collect(LinRange(0, 3*τ*1e6, 3*nτ));
            idx_w = convert(Int64, round(idx_w_f*nτ));

            # set the bounds for analysis
            idx1 = 2*nτ-idx_w # max(2*nτ-idx_w, 1)
            idx2 = 2*nτ+idx_w #min(2*nτ+idx_w, length(M))
            M_tar = broadcast(abs, M[idx1:idx2])
            t_tar = t[idx1:idx2]

            # initialize variables
            on_sweep = true
            curr_min = 0
            min_idx = 1
            min_idx_list = []

            # find the dips
            for idx = 1:length(M_tar)
                vh = M_tar[idx]
                if on_sweep # in a region of a dip, trying to find the minimum
                    if vh < curr_min
                        curr_min = vh
                        min_idx = idx
                    elseif vh > min_cutoff
                        push!(min_idx_list, min_idx)
                        curr_min = min_cutoff
                        on_sweep = false
                    end
                else # looking for next dip
                    if vh < min_cutoff
                        curr_min = vh
                        min_idx = idx
                        on_sweep = true
                    end
                end
            end

            # the last point is always a "dip"
            push!(min_idx_list, length(M_tar))

            # initialize stats arrays
            max_idx_list = [] # location of maxima
            max_vs = [] # height of maximum
            max_sigmas = [] # second moment of peak
            max_symms = [] # asymmetry factor of peak

            # go through the regions between dips and calculate stats
            for idx = 1:length(min_idx_list)-1

                # set the bounds
                tar_idx = min_idx_list[idx]:min_idx_list[idx+1]

                # calculate the height and location (relative to bounds) of the peak
                max_v, max_idx = findmax(M_tar[tar_idx]);

                # calculate the location of peak (absolute)
                max_idx = max_idx + tar_idx[1] - 1

                # if the peak is too small, don't calculate stats for it ie skip the rest of this iteration
                if max_v < 1.5*min_cutoff
                    continue
                end

                # set bounds around the peak for calculating stats
                tar_idx = max(max_idx - mean_range, min_idx_list[idx]):min(max_idx + mean_range, min_idx_list[idx+1])

                # set the window to calculate the symmetry
                symm_dist = min(sym_w, max_idx-1, length(M_tar)-max_idx)
                left_idx = max_idx-symm_dist:max_idx-1
                right_idx = max_idx+1:max_idx+symm_dist

                # calculate sigma
                sigma_h = sqrt( sum(M_tar[tar_idx].*(t_tar[tar_idx] .- t_tar[max_idx]).^2)./sum(M_tar[tar_idx]) )

                # calculate asymmtery
                lM = M_tar[left_idx];
                rM = reverse(M_tar[right_idx]);
                if rmse
                    symm_h = sqrt(sum((lM-rM).^2)/(length(lM)+length(rM)))
                else
                    symm_h = sum(lM.*rM)/sqrt(sum(lM.*lM)*sum(rM.*rM))
                end

                # save all the stats
                push!(max_idx_list,max_idx)    
                push!(max_sigmas,sigma_h)
                push!(max_vs,max_v)
                push!(max_symms,symm_h)

            end

            # write the results to a dictionary
            dict_here = Dict("min_idx"=>min_idx_list, "max_idx"=>max_idx_list,
                                "sigmas"=>max_sigmas, "vals"=>max_vs, "symms"=>max_symms,
                                "M"=>M_tar, "t"=>t_tar, "τ"=>τ, "nτ"=>nτ)

            # save the dictionary of results to a big dictionary (indexed by I)
            M_stats[string(i)] = dict_here

        end

        return M_stats

    end

## MAKE THE FUN PLOT FOR A SLICE OF THE DATA

    function make_Mstats_options(dat, var_key, var_units, dummy_idx, subplot_idxs)
    
        f = Dict()
        f["fsize"] = 9;
        f["rnd_digits"] = 3;
        f["subplot_colors"] = [:orange,:lightblue,:purple]
        f["size"] = (800, 400)
    
        # set the variable to plot vs
        f["var"] = dat[var_key];
        if var_key == "τ"
            f["var"] = f["var"]*1e6
        end
        f["var_idx"] = findall(y -> y == var_key, dat["vars"])[1]
        f["var_label"] = var_key*" = ";
        f["var_units"] = var_units;
        f["var_key"] = var_key;
    
        # set the tick marks for the axes on p1
        var_idx = f["var_idx"]
        var = f["var"]
        if var_key == "τ"
            c_τ = round(2*maximum(var))
            r_τ = round(c_τ/2);
            step = var[2]-var[1];
            f["t_ticks"] = round.((var[1]-step):(c_τ+r_τ)/5:(c_τ+r_τ));
        else
            τ_idx = findall(y -> y == "τ", dat["vars"])[1]
            if τ_idx > var_idx
                τ_idx -= 1;
            end
            τ_idx = dummy_idx[τ_idx];
            c_τ = round(2*1e6*dat["τ"][τ_idx]);
            r_τ = round(c_τ/2)
            f["t_ticks"] = round.((c_τ-r_τ):2*r_τ/5:(c_τ+r_τ))
        end
        f["M_ticks"] = 0:0.25:0.5
        f["var_ticks"] = round.(var[1]:(var[end]-var[1])/5:var[end], digits = f["rnd_digits"])
    
        # some labels
        f["p1_tlabel"] = "t (μs)"
        f["p3_tlabel"] = "t (μs)"
        f["subplot_Mlabel"] = "M"
    
        # placement info for some annotations
        f["subplot_note_y"] = 0.35;
        var_ticks = f["var_ticks"];
        t_ticks = f["t_ticks"];
        f["p1note"] = ("pre-echo", "post-echo")
        if var_key != "τ"
            f["p1note_x"] = (var_ticks[1]+0.2*(var_ticks[2]-var_ticks[1]), var_ticks[1]+0.2*(var_ticks[2]-var_ticks[1]))
            f["p1note_y"] = (t_ticks[2], t_ticks[end-1])
            f["p1note_angle"] = (90, 90)
        else
            f["p1note_x"] = (var_ticks[4], var_ticks[4])
            f["p1note_y"] = (t_ticks[2]/4, t_ticks[end-1])
            f["p1note_angle"] = (0, 0)
        end

        return f
    
    end

    function make_Mstats_plot(dat, M_stats, subplot_idxs, d_idx, dummy_idx, options)

        # load variable settings
        var = options["var"]
        var_key = options["var_key"]
        var_idx = options["var_idx"]
        var_label = options["var_label"]
        var_units = options["var_units"];
        
        # load axis ticks
        t_ticks = options["t_ticks"]
        M_ticks = options["M_ticks"]
        var_ticks = options["var_ticks"]

        # set the limits for the axes on p1
        var_lims = (var_ticks[1], var_ticks[end])
        t_lims = (t_ticks[1], t_ticks[end])
        M_lims = (M_ticks[1], M_ticks[end])
    
        # set the colors of the subplot traces
        subplot_colors = options["subplot_colors"]

        # something about resolution, i think?
        gr(dpi=100)
    
        # load a couple other things
        rnd_digits = options["rnd_digits"]
        p1_tlabel = options["p1_tlabel"]
        p3_tlabel = options["p3_tlabel"]
        x_loc = options["p1note_x"]
        y_loc = options["p1note_y"]
        ang = options["p1note_angle"]
        p1note = options["p1note"]
        subplot_Mlabel = options["subplot_Mlabel"]

        # create the subplot for the M_stats data (t vs var)
        p1 = plot(legend=:none, framestyle=:box, xticks=var_ticks, yticks=t_ticks, xlims=var_lims, ylims=t_lims);

        # plot vertical lines at the subplot variable locations
        for i = 1:length(subplot_idxs)
            plot!(p1, [var[subplot_idxs[i]]], seriestype=:vline, label="", color=subplot_colors[i], lw=2)
        end

        # create the three other subplots
        p_list = (plot(), plot(), plot());
        p_idx = 1;
    
        idx_list, d = make_idx(dat["vars"], dat)

        # do the plotting
        for i in idx_list

            # make sure you're in the right slice of the data
            dummy = [];
            for idx = 1:length(i)
                if idx != var_idx
                    push!(dummy, i[idx])
                end
            end

            if dummy != dummy_idx
                continue
            end

            # load the stats
            var_h = var[i[var_idx]] # value of variable being plotted vs
            stats_h = M_stats[string(i)]
            min_idx = stats_h["min_idx"] # location of dips
            max_idx = stats_h["max_idx"] # location of peaks between dips
            sigmas = stats_h["sigmas"] # second moment of peaks
            vals = stats_h["vals"] # height of peaks
            symms = stats_h["symms"] # asymmetry of peaks
            M = stats_h["M"]
            t = stats_h["t"]
            τ = stats_h["τ"]
            nτ = stats_h["nτ"]

            # set the time axes for the subplots
            t_ticks_h = round.(t[1]:(t[end]-t[1])/5:t[end])
            t_lims_h = (t_ticks_h[1], t_ticks_h[end])

            # loop over the number of maxima
            for idx = 1:length(max_idx)

                # set the transparency and size based on the M_stats data
                transp_h = 2*(0.5-vals[idx])
                fill_h = RGB(transp_h, transp_h, transp_h)
                color_h = RGB(0,0,0)
                msize = 30*sqrt(vals[idx]/sigmas[idx])

                # plot blobs for each peak
                scatter!(p1, [var_h], [t[max_idx[idx]]], color=fill_h, label="", markersize=msize, alpha=5*(1-transp_h))

            end

            # check if this is the iteration of one of the sample echoes
            if i[var_idx] in subplot_idxs
            
                # check for custom annotation placement
                if haskey(options, "subplot_note_x")
                    x_loc_h = options["subplot_note_x"]
                else
                    x_loc_h = t_ticks_h[end-1]
                end
                y_loc_h = options["subplot_note_y"]
                
                # load the subplot
                pt = p_list[p_idx];
             
                # write the annotation
                note = var_label*string(round(var_h, digits = rnd_digits))*var_units
            
                # plot the echo
                plot!(pt, t, M, lw=2, label="", xticks = t_ticks_h, yticks = M_ticks, xlims = t_lims_h, ylims = M_lims,
                    color = subplot_colors[p_idx], framestyle = :box)
                
                # annotate
                annotate!(pt, x_loc_h, y_loc_h, text(note, options["fsize"]+3))
            
                # plot the dip locations
                for idx in min_idx
                    plot!(pt, [t[idx]], seriestype=:vline, color=RGB(0,0,0), linestyle=:dash, label="") 
                end
            
                p_idx += 1
            
            end

        end

        xlabel!(p1, var_key)
        ylabel!(p1, p1_tlabel)
        annotate!(p1, x_loc[1], y_loc[1], text(p1note[1], rotation=ang[1]))
        annotate!(p1, x_loc[2], y_loc[2], text(p1note[2], rotation=ang[2]))
    
        if var_key == "τ"
            # plot the location of the expected echo for varying tau
            plot!(var, 2*var, linestyle=:dash, color=RGB(0,0,0))
        end

        for i = 1:3
            ylabel!(p_list[i], subplot_Mlabel)
        end
        xlabel!(p_list[3], p3_tlabel)

        l = @layout [a{0.5w} grid(3,1)]

        plt = plot(p1, p_list[1], p_list[2], p_list[3], layout = l, size = options["size"])

        return plt

    end

    export getMstats, make_Mstats_plot, make_Mstats_options

end