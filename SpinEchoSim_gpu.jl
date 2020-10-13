module SpinEchoSim

  include("gpu/SpinSimParams.jl")
  export SpinSimParams
  include("gpu/spin_sims.jl")
  export spin_echo_sim, spin_echo_sim_liouville, spin_echo_sim_liouville_cpmg
  include("lib/MTools.jl")
  export MTools

end

using .SpinEchoSim
using .SpinSimParams
using .MTools
