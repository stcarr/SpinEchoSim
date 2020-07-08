module MTools

    include("SpinSimParams.jl")
    using .SpinSimParams

    function getMstats(dat,idx_w=200,sym_w=50,min_cutoff=0.02,mean_range=50,t_max=300)

        # idx_w :  number of time indices on each side to keep
        # sym_w : number of time indices on each side to evaluate symmetry
        # min_cutoff : value of M to assess as a "dip"
        # mean_range : number of time indices on each side for evaluating the spread
    
        I, d = make_idx(dat["vars"], dat)
        M_list = dat["M_list"];
    
        # fix the "unwrapping" of multi-dim arrays by JSON
        nvars = length(dat["vars"])
        for idx = 1:nvars-1
            M_list = hcat(M_list...)
        end
    
        M_stats = Dict();

        for i in I
            M = [ x["re"] + im*x["im"] for x in M_list[i] ];
            nτ = convert(Int64, round(dat["τ"]*dat["γ"]/dat["dt"]));
            t = LinRange(0, t_max, size(M, 1));


            idx1 = max(2*nτ-idx_w,1)
            idx2 = min(2*nτ+idx_w,length(M))
            M_tar = broadcast(abs,M[idx1:idx2])
            t_tar = t[idx1:idx2]

            on_sweep = true
            curr_min = 0;
            min_idx = 1
            min_idx_list = []

            for idx = 1:length(M_tar)
                #dh = M_diff[idx]
                vh = M_tar[idx]
                if on_sweep # in a region of a dip
                    if vh < curr_min
                        curr_min = vh
                        min_idx = idx
                    elseif vh > min_cutoff
                        push!(min_idx_list,min_idx)
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

            push!(min_idx_list,length(M_tar))

            max_idx_list = []
            max_vs = []
            max_sigmas = []
            max_symms = []
            for i = 1:length(min_idx_list)-1

                tar_idx = min_idx_list[i]:min_idx_list[i+1]
                max_v, max_idx = findmax(M_tar[tar_idx]);
                max_idx = max_idx+tar_idx[1]-1

                if max_v < 1.5*min_cutoff
                    continue
                end

                tar_idx = max(max_idx-mean_range,min_idx_list[i]):min(max_idx+mean_range,min_idx_list[i+1])

                symm_dist = min(sym_w,max_idx-1,length(M_tar)-max_idx)
                left_idx = max_idx-symm_dist:max_idx-1
                right_idx = max_idx+1:max_idx+symm_dist

                #mean_h = sum(tar_idx.*M_tar[tar_idx])/sum(M_tar[tar_idx])

                sigma_h = sqrt( sum(M_tar[tar_idx].*(t_tar[tar_idx] .- t_tar[max_idx]).^2)./sum(M_tar[tar_idx]) )
                lM = M_tar[left_idx];
                rM = reverse(M_tar[right_idx]);
                symm_h = sum(lM.*rM)/sqrt(sum(lM.*lM)*sum(rM.*rM))

                push!(max_idx_list,max_idx)    
                push!(max_sigmas,sigma_h)
                push!(max_vs,max_v)
                push!(max_symms,symm_h)

            end

            dict_here = Dict("min_idx"=>min_idx_list,"max_idx"=>max_idx_list,
                                "sigmas"=>max_sigmas,"vals"=>max_vs,"symms"=>max_symms,
                                "M"=>M_tar, "t"=>t_tar)
            M_stats[string(i)] = dict_here

        end
    
        return M_stats
    
    end

    export getMstats

end