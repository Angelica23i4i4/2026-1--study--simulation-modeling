using DrWatson
@quickactivate "project"

using ResumableFunctions
using ConcurrentSim
using Distributions
using StableRNGs
using Statistics
using CairoMakie

# ===== Параметры =====
const RUNS   = 5
const N      = 10
const S      = 3
const LAMBDA = 100
const MU     = 1

const rng = StableRNG(42)
const F   = Exponential(LAMBDA)
const G   = Exponential(MU)

# ===== Код из лабораторной =====
@resumable function machine(
    env::Environment,
    repair_facility::Resource,
    spares::Store{Process},
)
    while true
        try
            @yield timeout(env, Inf)
        catch
        end
        @yield timeout(env, rand(rng, F))
        get_spare = take!(spares)
        @yield get_spare | timeout(env)
        if state(get_spare) != ConcurrentSim.idle
            @yield interrupt(value(get_spare))
        else
            throw(StopSimulation("No more spares!"))
        end
        @yield request(repair_facility)
        @yield timeout(env, rand(rng, G))
        @yield unlock(repair_facility)
        @yield put!(spares, active_process(env))
    end
end

@resumable function start_sim(
    env::Environment,
    repair_facility::Resource,
    spares::Store{Process},
    n::Int,
    s::Int,
)
    for i in 1:n
        proc = @process machine(env, repair_facility, spares)
        @yield interrupt(proc)
    end
    for i in 1:s
        proc = @process machine(env, repair_facility, spares)
        @yield put!(spares, proc)
    end
end

function sim_repair(; n=N, s=S, num_repairmen=1)
    sim             = Simulation()
    repair_facility = Resource(sim, num_repairmen)
    spares          = Store{Process}(sim)
    @process start_sim(sim, repair_facility, spares, n, s)
    msg       = run(sim)
    stop_time = now(sim)
    println("At time $stop_time: $msg")
    return stop_time
end

# ===== Базовый прогон (как в лабораторной) =====
println("=" ^ 40)
println("Базовый прогон N=$N, S=$S, ремонтников=1")
results = Float64[]
for i in 1:RUNS
    push!(results, sim_repair())
end
println("Average crash time: ", sum(results) / RUNS)

# ===== Прогон для разного N =====
println("\nРазное количество машин N...")
n_range = [5, 10, 15, 20]
means_N = Float64[]
for n_val in n_range
    times = [sim_repair(n=n_val) for _ in 1:RUNS]
    push!(means_N, mean(times))
    println("  N=$n_val => E[T] = $(round(mean(times), digits=2)) ч.")
end

# ===== Прогон для разного числа ремонтников =====
println("\nРазное число ремонтников...")
rep_range = [1, 2, 3]
means_R   = Float64[]
for r in rep_range
    times = [sim_repair(num_repairmen=r) for _ in 1:RUNS]
    push!(means_R, mean(times))
    println("  Ремонтников=$r => E[T] = $(round(mean(times), digits=2)) ч.")
end

# ===== Аналитика =====
function analytical_ET(n, s, lam, mu)
    total = n + s
    m     = s + 1
    A     = zeros(m, m)
    b     = ones(m)
    for (idx, i) in enumerate(n:total)
        lam_rate    = min(n, i) / lam
        mu_rate     = (i < total) ? 1.0 / mu : 0.0
        A[idx, idx] = lam_rate + mu_rate
        idx < m && (A[idx, idx+1] -= mu_rate)
        idx > 1 && (A[idx, idx-1] -= lam_rate)
    end
    return (A \ b)[end]
end

mean_ana = analytical_ET(N, S, LAMBDA, MU)
println("\nАналитика E[T]: $(round(mean_ana, digits=2)) ч.")
println("Симуляция E[T]: $(round(mean(results), digits=2)) ч.")

# ===== Графики =====
mkpath(plotsdir())

# График 1: базовые прогоны + сравнение с аналитикой
fig1 = Figure(size=(700, 400))
ax1  = Axis(fig1[1,1],
    xlabel = "Прогон",
    ylabel = "Время до краша (ч.)",
    title  = "Модель Росса: N=$N, S=$S, ремонтников=1")
barplot!(ax1, 1:RUNS, results, color=:steelblue)
hlines!(ax1, [mean(results)], color=:red,    linewidth=2, label="Среднее (сим.)")
hlines!(ax1, [mean_ana],      color=:orange, linewidth=2, linestyle=:dash, label="Аналитика")
axislegend(ax1)
save(plotsdir("ross_base.png"), fig1)
println("Сохранён: ross_base.png")

# График 2: E[T] vs число машин N
ana_N = [analytical_ET(n, S, LAMBDA, MU) for n in n_range]
fig2  = Figure(size=(700, 400))
ax2   = Axis(fig2[1,1],
    xlabel = "Число машин N",
    ylabel = "E[T] до краша (ч.)",
    title  = "Модель Росса: влияние числа машин")
lines!(ax2,   n_range, ana_N,   color=:red,       linewidth=2,   label="Аналитика")
scatter!(ax2, n_range, means_N, color=:steelblue, markersize=12, label="Симуляция")
axislegend(ax2)
save(plotsdir("ross_vs_N.png"), fig2)
println("Сохранён: ross_vs_N.png")

# График 3: E[T] vs число ремонтников
fig3 = Figure(size=(700, 400))
ax3  = Axis(fig3[1,1],
    xlabel = "Число ремонтников",
    ylabel = "E[T] до краша (ч.)",
    title  = "Модель Росса: влияние ремонтников")
barplot!(ax3, rep_range, means_R, color=:mediumseagreen)
save(plotsdir("ross_vs_repairmen.png"), fig3)
println("Сохранён: ross_vs_repairmen.png")

println("\nГотово! Графики в папке plots/")
