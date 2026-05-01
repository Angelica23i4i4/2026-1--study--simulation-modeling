using DrWatson
@quickactivate "project"
include(srcdir("SIRPetri.jl"))
using .SIRPetri
using DataFrames, Plots

β = 0.3
γ = 0.1
tmax = 100.0

net, u0, _ = build_sir_network(β, γ)
df = simulate_deterministic(net, u0, (0.0, tmax), saveat = 0.2, rates = [β, γ])

anim = @animate for i in 1:nrow(df)
    bar(
        ["S", "I", "R"],
        [df.S[i], df.I[i], df.R[i]],
        ylim = (0, 1050),
        xlabel = "Compartment",
        ylabel = "Population",
        title = "SIR dynamics at t = $(round(df.time[i], digits=1))",
        legend = false,
        color = [:blue :red :green],
    )
end

gif(anim, plotsdir("sir_animation.gif"), fps = 20)
println("Анимация сохранена.")
