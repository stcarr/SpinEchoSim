function lorentzian(x, μ, Γ)
    L = (1/π)*(Γ/2)/((x-μ)^2+(Γ/2)^2)
    return L
end

function gaussian(x,μ,σ)
    G = (1/(σ*sqrt(2*pi)))*exp(-(1/2)*((x.-μ)/σ).^2)
    return G
end
