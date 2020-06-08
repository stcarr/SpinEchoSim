mutable struct SpinEnsemble
        
    ν0::Float64 # central frequnecy
    P::Array{Float64,1} # list spin weights
    ν::Array{Float64,1} # list of spin frequencies    
    
    #ρ_init::Array{SMatrix{2,2,Float64},2} # initial spins
    ρ_init
    
    function SpinEnsemble(ν0, P, ν, ρ_init)
        new(ν0, P, ν, ρ_init)
    end
end