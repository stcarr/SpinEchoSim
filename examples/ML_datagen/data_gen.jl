using DelimitedFiles

# echoes are abou 34 kB to store in JLD2 format
# probably want about 100,000 samples
# (will give 400 MB of echos)

num_samples = 10;

alpha_list = 0.2*rand(num_samples,1);
xi_list = 20.0*rand(num_samples,1);
p_list = 2.0 .+ 2.0*rand(num_samples,1);
d_list = 3.0 .+ 3.0*rand(num_samples,1);

dat_array = hcat(alpha_list, xi_list, p_list, d_list)

fname = "inputs.txt"

open(fname,"w") do io
    writedlm(io,dat_array)
end
