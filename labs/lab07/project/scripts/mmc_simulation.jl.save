using DrWatson
@quickactivate "project"

using StableRNGs
using Distributions
using ConcurrentSim
using ResumableFunctions
using CairoMakie  # используй CairoMakie вместо Plots (совместимо с Julia 1.12)
using Statistics

# ===== Параметры симуляции =====
const NUM_CUSTOMERS = 500
const NUM_SERVERS   = 2
const MU            = 1.0 / 2   # интенсивность обслуживания
const LAM           = 0.9       # интенсивность входящего потока
const SEED          = 123

# ===== Поведение клиента =====
@resumable function customer(
    env::Environment,
    server::Resource,
    id::Integer,
    t_a::Float64,
    d_s::Distribution,
    rng,
    wait_times::Vector{Float64},
)
    @yield timeout(env, t_a)          # клиент прибывает
    t_arrive = now(env)
    @yield request(server)            # клиент занимает сервер
    push!(wait_times, now(env) - t_arrive)  # фиксируем время ожидания
    @yield timeout(env, rand(rng, d_s))     # обслуживание
    @yield unlock(server)             # клиент уходит
end

# ===== Запуск одной симуляции =====
function run_mmc(;
    num_servers   = NUM_SERVERS,
    mu            = MU,
    lam           = LAM,
    num_customers = NUM_CUSTOMERS,
    seed          = SEED,
)
    rng          = StableRNG(seed)
    arrival_dist = Exponential(1 / lam)
    service_dist = Exponential(1 / mu)
    wait_times   = Float64[]

    sim    = Simulation()
    server = Resource(sim, num_servers)

    arrival_time = 0.0
    for i in 1:num_customers
        arrival_time += rand(rng, arrival_dist)
        @process customer(sim, server, i, arrival_time, service_dist, rng, wait_times)
    end

    run(sim)
    return wait_times
end

# ===== Аналитические формулы M/M/c (формула Эрланга) =====
function analytical_mmc(lam, mu, c)
    rho = lam / (c * mu)
    rho >= 1 && return (NaN, NaN, NaN)

    sum_part = sum((c * rho)^n / factorial(n) for n in 0:(c-1))
    P0 = 1.0 / (sum_part + (c * rho)^c / (factorial(c) * (1 - rho)))

    P_wait = ((c * rho)^c / (factorial(c) * (1 - rho))) * P0
    Lq     = rho / (1 - rho) * P_wait
    Wq     = Lq / lam
    W      = Wq + 1 / mu
    return (Wq, W, Lq)
end

# ===== Основной прогон =====
wt = run_mmc()

Wq_theory, W_theory, Lq_theory = analytical_mmc(LAM, MU, NUM_SERVERS)

println("=== Результаты M/M/c ===")
println("Среднее время ожидания (симуляция):  ", round(mean(wt), digits=4))
println("Среднее время ожидания (аналитика):  ", round(Wq_theory, digits=4))
println("Доля клиентов без ожидания:           ", round(count(==(0.0), wt) / length(wt), digits=4))

# ===== Графики =====
mkpath(plotsdir())

# 1. Гистограмма времён ожидания
fig1 = Figure(size = (800, 400))
ax1  = Axis(fig1[1, 1],
    xlabel = "Время ожидания",
    ylabel = "Число заявок",
    title  = "M/M/c: распределение времени ожидания")
hist!(ax1, wt, bins = 40, color = (:steelblue, 0.7))
vlines!(ax1, [mean(wt)], color = :red, linewidth = 2, label = "Среднее (сим.)")
vlines!(ax1, [Wq_theory], color = :orange, linewidth = 2, linestyle = :dash, label = "Аналитика")
axislegend(ax1)
save(plotsdir("mmc_histogram.png"), fig1)
println("Сохранено: mmc_histogram.png")

# 2. Накопленное среднее время ожидания
fig2 = Figure(size = (800, 400))
ax2  = Axis(fig2[1, 1],
    xlabel = "Номер заявки",
    ylabel = "Накопленное среднее ожидания",
    title  = "M/M/c: сходимость среднего")
cumavg = cumsum(wt) ./ (1:length(wt))
lines!(ax2, 1:length(wt), cumavg, color = :steelblue, label = "Симуляция")
hlines!(ax2, [Wq_theory], color = :red, linewidth = 2, linestyle = :dash, label = "Аналитика")
axislegend(ax2)
save(plotsdir("mmc_cumavg.png"), fig2)
println("Сохранено: mmc_cumavg.png")

# 3. Сравнение для разных значений LAM
lam_range = 0.1:0.1:0.95
sim_means = Float64[]
ana_means = Float64[]

for l in lam_range
    wt_l = run_mmc(lam = l)
    push!(sim_means, mean(wt_l))
    Wq, _, _ = analytical_mmc(l, MU, NUM_SERVERS)
    push!(ana_means, Wq)
end

fig3 = Figure(size = (800, 400))
ax3  = Axis(fig3[1, 1],
    xlabel = "Интенсивность входящего потока λ",
    ylabel = "Среднее время ожидания Wq",
    title  = "M/M/c: симуляция vs аналитика")
scatter!(ax3, collect(lam_range), sim_means, color = :steelblue, label = "Симуляция")
lines!(ax3,   collect(lam_range), ana_means,  color = :red, linewidth = 2, label = "Аналитика")
axislegend(ax3)
save(plotsdir("mmc_lam_sweep.png"), fig3)
println("Сохранено: mmc_lam_sweep.png")
