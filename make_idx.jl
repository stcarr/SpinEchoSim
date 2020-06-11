include("make_parameters.jl")

function make_idx(vars, params)
    
    fields = string.(fieldnames(make_parameters));

    # create the tuple with the dimensions needed
    d = ();
    p = 1;
    for x in intersect(vars, fields)
        f = getproperty(params, Symbol(x));
        lf = size(f,1)
        p = p*lf;
        d = (d..., lf);
    end
    f = sample(1:p, d, replace = false)

    # fill it with indices
    I = [findall(x -> x == temp, f)[1] for temp in f]
    
    return I, d
end