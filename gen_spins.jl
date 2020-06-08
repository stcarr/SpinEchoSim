using Random, Distributions
include("custom_dists.jl")

function gen_spins(ν0,nS,int_sample=true,dist_type=0,width=0.5)

    μ = ν0; # mean freq, 10 MHZ
    Γ = 0.05; # width of Lorentz/Cauchy
    σ = 0.25; # std of Gaussian/Normal
    
    ν_full = zeros(nS)
    P_full = zeros(nS)
    
    if (int_sample == true)
        ν_full = collect(LinRange(μ-width/2.0,μ+width/2.0, nS));
        if dist_type == 0
            P_full = gaussian.(ν_full, μ, σ);
        else
            P_full = lorentzian.(ν_full, μ, Γ);
        end
        
        P_full = P_full/sum(P_full) # normalize;
        return ν_full, P_full
    else
        if dist_type == 0
            d = Normal(μ,σ)
        else
            d = Cauchy(10,Γ*2.0)            
        end
        for v = 1:nS
            v_here = rand(d)
            while (abs(v_here - μ) > width)
                v_here = rand(d)
            end
            ν_full[v] = v_here; # random samples
        end
        P_full = ones(nS)./nS; # uniform probability
        return ν_full, P_full        
    end


    
end