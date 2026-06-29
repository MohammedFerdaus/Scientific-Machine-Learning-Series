using LinearAlgebra
using Statistics

function build_system_matrix(omega)
    return [0 1; -omega^2 0]
end

function oscillator_deriv(state, system_matrix)
    return system_matrix * state  # was system_matrixv 
end

function rk4_step(state, system_matrix, dt)
    k1 = oscillator_deriv(state, system_matrix)
    k2 = oscillator_deriv(state + dt/2 * k1, system_matrix)
    k3 = oscillator_deriv(state + dt/2 * k2, system_matrix)
    k4 = oscillator_deriv(state + dt * k3, system_matrix)
    
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
end

function run_simulation(x0, v0, omega, dt, num_steps)
    times = zeros(Float64, num_steps + 1)
    positions = zeros(Float64, num_steps + 1)
    velocities = zeros(Float64, num_steps + 1)

    times[1] = 0.0
    positions[1] = x0
    velocities[1] = v0

    system_matrix = build_system_matrix(omega)
    for i in 1:num_steps
        state = [positions[i], velocities[i]]
        next_state = rk4_step(state, system_matrix, dt)
        times[i + 1] = times[i] + dt
        positions[i + 1] = next_state[1]
        velocities[i + 1] = next_state[2]
    end

    return (times, positions, velocities) 
end

function exact_solution(t, x0, v0, omega)
    amplitude = sqrt(x0^2 + (v0 / omega)^2) 
    phase = atan(-v0 / omega, x0)
    return amplitude * cos(omega * t + phase)
end

function compute_errors(positions, times, x0, v0, omega)
    exact_positions = [exact_solution(t, x0, v0, omega) for t in times]
    pointwise_errors = abs.(positions - exact_positions)
    max_error = maximum(pointwise_errors)
    rms_error = sqrt(mean(pointwise_errors .^ 2))
    return (max_error, rms_error)
end
