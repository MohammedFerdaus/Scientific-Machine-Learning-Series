include("numerical_version.jl")
include("pinn_train.jl")
include("pinn_train_autograd.jl")

using CairoMakie

omega = 1.0
x0 = 1.0
v0 = 0.0
dt = 0.01
num_steps = 1000
time_span = num_steps * dt

function test_numerical(omega, x0, v0, dt, num_steps)
    times, positions, velocities = run_simulation(x0, v0, omega, dt, num_steps)
    (max_error, rms_error) = compute_errors(positions, times, x0, v0, omega)
    
    println("Numerical Results")
    println("Maximum Absolute Error (Max Error): ", max_error)
    println("Root Mean Square Error (RMS Error): ", rms_error)
    
    @assert max_error < 1e-6 "Validation failed: Simulation drift ($max_error) exceeds the 1e-6 limit!"
    
    return (times, positions, velocities)
end

function test_pinn(omega, time_span, times)
    trained_params = train(omega, time_span, 50, 100, 1e-3, 10.0, [1,16,16,1], 42)

    normalized_times = times ./ time_span
    pinn_positions = map(t -> u_fn(t, trained_params), normalized_times)

    exact_positions = cos.(omega .* times)
    errors = abs.(pinn_positions .- exact_positions)
    max_error = maximum(errors)
    rms_error = sqrt(mean(errors .^ 2))

    println("PINN (Finite Difference) Results")
    println("Maximum Absolute Error (Max Error): ", max_error)
    println("Root Mean Square Error (RMS Error): ", rms_error)

    @assert max_error < 10.0 "Validation failed: PINN deviation ($max_error) exceeds limit"

    return pinn_positions
end

function test_pinn_autograd(omega, time_span, times)
    trained_params, loss_history = train_autograd(omega, time_span, 1000, 250, 2e-4, 1.0, [1,32,32,32,1], 42)

    normalized_times = times ./ time_span
    pinn_autograd_positions = map(t -> u_fn(t, trained_params), normalized_times)

    exact_positions = cos.(omega .* times)
    errors = abs.(pinn_autograd_positions .- exact_positions)
    max_error = maximum(errors)
    rms_error = sqrt(mean(errors .^ 2))

    println("Autograd PINN Results")
    println("Maximum Absolute Error (Max Error): ", max_error)
    println("Root Mean Square Error (RMS Error): ", rms_error)

    @assert max_error < 2.0 "Validation failed: Autograd PINN deviation ($max_error) exceeds limit"

    return (pinn_autograd_positions, loss_history)
end

function plot_numerical(times, positions, velocities, omega, x0, v0)
    fig = Figure(size=(1200, 400))
    
    exact_positions = x0 .* cos.(omega .* times) .+ (v0 / omega) .* sin.(omega .* times)
    exact_velocities = -x0 * omega .* sin.(omega .* times) .+ v0 .* cos.(omega .* times)
    errors = abs.(positions .- exact_positions)

    ax1 = Axis(fig[1, 1], title="Trajectories", xlabel="Time", ylabel="Position")
    lines!(ax1, times, positions, label="Numerical (RK4)", color=:blue)
    lines!(ax1, times, exact_positions, label="Exact", color=:red, linestyle=:dash)
    axislegend(ax1)

    ax2 = Axis(fig[1, 2], title="Absolute Position Error", xlabel="Time", ylabel="Error")
    lines!(ax2, times, errors, color=:blue)

    ax3 = Axis(fig[1, 3], title="Phase Portrait", xlabel="Position", ylabel="Velocity")
    lines!(ax3, positions, velocities, label="Numerical", color=:blue)
    lines!(ax3, exact_positions, exact_velocities, label="Exact", color=:red, linestyle=:dash)
    axislegend(ax3)

    save("numerical_results.png", fig)
end

function plot_comparison(times, numerical_positions, pinn_positions, pinn_autograd_positions, omega)
    fig = Figure(size=(1400, 800))

    exact_positions = cos.(omega .* times)

    ax1 = Axis(fig[1, 1], title="Trajectory comparison", xlabel="Time", ylabel="Position")
    lines!(ax1, times, exact_positions, label="Exact", color=:red, linestyle=:dash)
    lines!(ax1, times, numerical_positions, label="RK4", color=:blue)
    lines!(ax1, times, pinn_positions, label="PINN (FD)", color=:green)
    lines!(ax1, times, pinn_autograd_positions, label="PINN (Autograd)", color=:purple)
    axislegend(ax1)

    ax2 = Axis(fig[1, 2], title="Absolute error comparison", xlabel="Time", ylabel="Error")
    lines!(ax2, times, abs.(numerical_positions .- exact_positions), label="RK4", color=:blue)
    lines!(ax2, times, abs.(pinn_positions .- exact_positions), label="PINN (FD)", color=:green)
    lines!(ax2, times, abs.(pinn_autograd_positions .- exact_positions), label="PINN (Autograd)", color=:purple)
    axislegend(ax2)

    save("comparison_results.png", fig)
end

(times, numerical_positions, velocities) = test_numerical(omega, x0, v0, dt, num_steps)
plot_numerical(times, numerical_positions, velocities, omega, x0, v0)

pinn_positions = test_pinn(omega, time_span, times)

(pinn_autograd_positions, loss_history) = test_pinn_autograd(omega, time_span, times)

plot_comparison(times, numerical_positions, pinn_positions, pinn_autograd_positions, omega)