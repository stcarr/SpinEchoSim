mutable struct EchoParams

    dt::Float64 # time step
    nτ::Int64 # number of discrete time steps
    
    ω::Float64 # operator frequency
    α::Float64 # operator strength
    
    #M_op::SMatrix{Float64,2} # Magnetiziation operator
    #Iz::SMatrix{Float64,2} # Spin time propagation matrix
    #UL90::SMatrix{Float64,2} # 90 degree phase operators
    #UR90::SMatrix{Float64,2}
    M_op
    Iz
    UL90
    UR90
    Lj_list
    
    function EchoParams(dt, nτ, ω, α, M_op, Iz, UL90, UR90)
        
        new(dt, nτ, ω, α, M_op, Iz, UL90, UR90, nothing)
    end
end