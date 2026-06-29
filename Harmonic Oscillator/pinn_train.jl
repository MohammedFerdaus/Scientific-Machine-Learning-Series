using LinearAlgebra
using Statistics
include("pinn_architecture.jl")

function physics_residual(t, params, omega, time_span)
    tau = t/time_span
    u_value = u_fn(tau, params)
    u_double_deriv = ddu_fn(tau, params)
    return u_double_deriv / time_span^2 + omega^2 * u_value
end

function ic_residual(params, time_span)
    position_residual = u_fn(0.0, params) - 1.0
    velocity_residual = du_fn(0.0, params) / time_span
    return (position_residual, velocity_residual)
end

function compute_loss(params, collocation_points, omega, time_span, ic_weight)
    physics_loss = mean(t -> physics_residual(t, params, omega, time_span)^2, collocation_points)  
    (position_residual, velocity_residual) = ic_residual(params, time_span)
    ic_loss = position_residual^2 + velocity_residual^2
    return physics_loss + ic_weight * ic_loss
end

function compute_gradients(params, collocation_points, omega, time_span, ic_weight)
    epsilon = 1e-5
    grad_params = Vector{Tuple{Matrix{Float64}, Vector{Float64}}}(undef, length(params))

    for layer_index in 1:length(params)
        (weights, biases) = params[layer_index]
        grad_weights = zeros(size(weights))
        grad_biases = zeros(size(biases))

        for i in eachindex(weights)
            weights[i] += epsilon
            loss_plus = compute_loss(params, collocation_points, omega, time_span, ic_weight)
            weights[i] -= 2*epsilon
            loss_minus = compute_loss(params, collocation_points, omega, time_span, ic_weight)
            weights[i] += epsilon
            grad_weights[i] = (loss_plus - loss_minus) / (2*epsilon)
        end

        for i in eachindex(biases)
            biases[i] += epsilon
            loss_plus = compute_loss(params, collocation_points, omega, time_span, ic_weight)
            biases[i] -= 2*epsilon
            loss_minus = compute_loss(params, collocation_points, omega, time_span, ic_weight)
            biases[i] += epsilon
            grad_biases[i] = (loss_plus - loss_minus) / (2*epsilon)
        end

        grad_params[layer_index] = (grad_weights, grad_biases)
    end
    return grad_params
end

function adam_step(params, grad_params, adam_state, learning_rate, step)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    for layer_index in 1:length(params)
        (weights, biases) = params[layer_index]
        (grad_weights, grad_biases) = grad_params[layer_index]
        (first_moment_weights, second_moment_weights,
         first_moment_biases, second_moment_biases) = adam_state[layer_index]

        first_moment_weights = beta1 * first_moment_weights + (1 - beta1) * grad_weights
        second_moment_weights = beta2 * second_moment_weights + (1 - beta2) * (grad_weights .^ 2)
        
        corrected_first_weights = first_moment_weights / (1 - beta1^step)
        corrected_second_weights = second_moment_weights / (1 - beta2^step)
        
        new_weights = weights .- learning_rate .* corrected_first_weights ./ (sqrt.(corrected_second_weights) .+ epsilon)

        first_moment_biases = beta1 * first_moment_biases + (1 - beta1) * grad_biases
        second_moment_biases = beta2 * second_moment_biases + (1 - beta2) * (grad_biases .^ 2)
        
        corrected_first_biases = first_moment_biases / (1 - beta1^step)
        corrected_second_biases = second_moment_biases / (1 - beta2^step)
        
        new_biases = biases .- learning_rate .* corrected_first_biases ./ (sqrt.(corrected_second_biases) .+ epsilon)

        adam_state[layer_index] = (first_moment_weights, second_moment_weights,
                                   first_moment_biases, second_moment_biases)
        params[layer_index] = (new_weights, new_biases)
    end 

    return (params, adam_state)
end

function train(omega, time_span, num_collocation, num_epochs, learning_rate, ic_weight, layer_sizes, random_seed)
    params = init_params(layer_sizes, random_seed)
    collocation_points = collect(range(0.0, time_span, length=num_collocation))
    num_layers = length(params)
    adam_state = Vector{Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Float64}}}(undef, num_layers)
    
    for layer_index in 1:num_layers
        (weights, biases) = params[layer_index]
        adam_state[layer_index] = (
            zeros(size(weights)), 
            zeros(size(weights)),
            zeros(size(biases)),  
            zeros(size(biases)))
    end

    for epoch in 1:num_epochs
        grad_params = compute_gradients(params, collocation_points, omega, time_span, ic_weight)
        (params, adam_state) = adam_step(params, grad_params, adam_state, learning_rate, epoch)
        
        current_loss = compute_loss(params, collocation_points, omega, time_span, ic_weight)
        println("Epoch: ", epoch, " | Loss: ", current_loss)
    end

    return params
end