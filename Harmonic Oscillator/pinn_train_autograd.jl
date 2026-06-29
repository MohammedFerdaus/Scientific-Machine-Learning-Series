using LinearAlgebra
using Statistics
include("autograd.jl")

function wrap_params(tape, params)
    num_layers = length(params)
    
    tracked_params = Vector{Tuple{Matrix{TrackedValue}, Vector{TrackedValue}}}(undef, num_layers)
    
    for layer_index in 1:num_layers
        (weights, biases) = params[layer_index]
        
        tracked_weights = Matrix{TrackedValue}(undef, size(weights))
        for i in eachindex(weights)
            tracked_weights[i] = track_param(tape, weights[i])
        end
        
        tracked_biases = Vector{TrackedValue}(undef, length(biases))
        for i in eachindex(biases)
            tracked_biases[i] = track_param(tape, biases[i])
        end
        
        tracked_params[layer_index] = (tracked_weights, tracked_biases)
    end
    
    return tracked_params
end

function tracked_ddu_fn(tape, tau, tracked_params, cache)
    num_layers = length(tracked_params)

    tracked_first_deriv = [track_constant(tape, 1.0)]
    tracked_second_deriv = [track_constant(tape, 0.0)]

    (output_weights, _) = tracked_params[num_layers]
    fan_in_output = size(output_weights, 2)
    
    out_first = Vector{TrackedValue}(undef, fan_in_output)
    out_second = Vector{TrackedValue}(undef, fan_in_output)
    
    for j in 1:fan_in_output
        out_first[j] = tracked_scale(tape, tracked_first_deriv[1], output_weights[1, j].value)
        out_second[j] = tracked_scale(tape, tracked_second_deriv[1], output_weights[1, j].value)
    end
    tracked_first_deriv = out_first
    tracked_second_deriv = out_second

    for layer_index in (num_layers - 1):-1:1
        (tracked_tanh_output, tracked_tanh_grad) = cache[layer_index]
        n_elements = length(tracked_first_deriv)
        
        layer_first = Vector{TrackedValue}(undef, n_elements)
        layer_second = Vector{TrackedValue}(undef, n_elements)
        
        for j in 1:n_elements
            tanh_second_j = tracked_scale(tape,
                tracked_multiply(tape, tracked_tanh_output[j], tracked_tanh_grad[j]), -2.0)
                
            layer_second[j] = tracked_add(tape,
                tracked_multiply(tape, tracked_second_deriv[j], tracked_tanh_grad[j]),
                tracked_multiply(tape, tracked_first_deriv[j], tanh_second_j))
                
            layer_first[j] = tracked_multiply(tape, tracked_first_deriv[j], tracked_tanh_grad[j])
        end
        tracked_first_deriv = layer_first
        tracked_second_deriv = layer_second

        (weights, _) = tracked_params[layer_index]
        fan_in = size(weights, 2)
        fan_out = size(weights, 1)
        
        new_first = Vector{TrackedValue}(undef, fan_in)
        new_second = Vector{TrackedValue}(undef, fan_in)
        
        for k in 1:fan_in
            acc_first = tracked_scale(tape, tracked_first_deriv[1], weights[1, k].value)
            acc_second = tracked_scale(tape, tracked_second_deriv[1], weights[1, k].value)
            
            for j in 2:fan_out
                acc_first = tracked_add(tape, acc_first, tracked_scale(tape, tracked_first_deriv[j], weights[j, k].value))
                acc_second = tracked_add(tape, acc_second, tracked_scale(tape, tracked_second_deriv[j], weights[j, k].value))
            end
            
            new_first[k] = acc_first
            new_second[k] = acc_second
        end
        tracked_first_deriv = new_first
        tracked_second_deriv = new_second
    end

    return first(tracked_second_deriv)
end

function tracked_forward(tape, tau, tracked_params)
    num_layers = length(tracked_params)
    tracked_activation = [track_constant(tape, tau)]
    cache = Vector{NamedTuple{(:tracked_tanh_output, :tracked_tanh_grad),
                   Tuple{Vector{TrackedValue}, Vector{TrackedValue}}}}(undef, num_layers - 1)

    for hidden_layer in 1:(num_layers - 1)
        (tracked_weights, tracked_biases) = tracked_params[hidden_layer]
        fan_out = size(tracked_weights, 1)
        next_activation = Vector{TrackedValue}(undef, fan_out)
        tracked_tanh_grad = Vector{TrackedValue}(undef, fan_out)

        for i in 1:fan_out
            weight_row = tracked_weights[i, :]
            pre_act = tracked_matmul(tape, weight_row, tracked_activation)
            pre_act = tracked_add(tape, pre_act, tracked_biases[i])
            tanh_out = tracked_tanh(tape, pre_act)
            tanh_sq = tracked_square(tape, tanh_out)
            tanh_gr = tracked_subtract(tape, track_constant(tape, 1.0), tanh_sq)
            next_activation[i] = tanh_out
            tracked_tanh_grad[i] = tanh_gr
        end
        cache[hidden_layer] = (tracked_tanh_output = next_activation,
                               tracked_tanh_grad = tracked_tanh_grad)
        tracked_activation = next_activation
    end

    (output_weights, output_biases) = tracked_params[num_layers]
    pre_act = tracked_matmul(tape, output_weights[1, :], tracked_activation)
    output = tracked_add(tape, pre_act, output_biases[1])

    return (output, cache)
end

function forward_loss(tape, tracked_params, collocation_points, omega, time_span, ic_weight)
    squared_residuals = Vector{TrackedValue}(undef, length(collocation_points))
    tracked_omega_sq = track_constant(tape, omega^2)
    epsilon = 1e-4

    for i in 1:length(collocation_points)
        tau = collocation_points[i] / time_span
        tau_plus = (collocation_points[i] + epsilon) / time_span
        tau_minus = (collocation_points[i] - epsilon) / time_span

        (u_value, _) = tracked_forward(tape, tau, tracked_params)
        (u_plus, _) = tracked_forward(tape, tau_plus, tracked_params)
        (u_minus, _) = tracked_forward(tape, tau_minus, tracked_params)

        two_u = tracked_scale(tape, u_value, 2.0)
        numerator = tracked_add(tape,
                        tracked_subtract(tape, u_plus, two_u),
                        u_minus)
        tracked_ddu_scaled = tracked_scale(tape, numerator, 1.0 / (epsilon^2 * time_span^2))

        omega_sq_u = tracked_multiply(tape, tracked_omega_sq, u_value)
        residual = tracked_add(tape, tracked_ddu_scaled, omega_sq_u)
        squared_residuals[i] = tracked_square(tape, residual)
    end

    physics_loss = tracked_mean(tape, squared_residuals)

    (u_at_zero, _) = tracked_forward(tape, 0.0, tracked_params)
    tracked_pos_residual = tracked_subtract(tape, u_at_zero, track_constant(tape, 1.0))
    tracked_vel_residual = track_constant(tape, 0.0)

    ic_loss = tracked_add(tape,
                  tracked_square(tape, tracked_pos_residual),
                  tracked_square(tape, tracked_vel_residual))

    tracked_ic_weight = track_constant(tape, ic_weight)
    weighted_ic_loss = tracked_multiply(tape, tracked_ic_weight, ic_loss)
    total_loss = tracked_add(tape, physics_loss, weighted_ic_loss)

    return total_loss
end

function compute_gradients_autograd(params, collocation_points, omega, time_span, ic_weight)
    tape = new_tape()
    tracked_params = wrap_params(tape, params)
    
    loss_tracked = forward_loss(tape, tracked_params, collocation_points,
                                omega, time_span, ic_weight)

    backward!(tape, loss_tracked)

    grad_params = Vector{Tuple{Matrix{Float64}, Vector{Float64}}}(undef, length(params))
    
    for layer_index in 1:length(params)
        (tracked_weights, tracked_biases) = tracked_params[layer_index]

        grad_weights = Matrix{Float64}(undef, size(tracked_weights))
        for i in eachindex(tracked_weights) 
            grad_weights[i] = extract_grad(tape, tracked_weights[i])
        end

        grad_biases = Vector{Float64}(undef, length(tracked_biases))
        for i in eachindex(tracked_biases) 
            grad_biases[i] = extract_grad(tape, tracked_biases[i])
        end

        grad_params[layer_index] = (grad_weights, grad_biases)
    end
    
    return grad_params
end

function train_autograd(omega, time_span, num_collocation, num_epochs, learning_rate, ic_weight, layer_sizes, random_seed)
    params = init_params(layer_sizes, random_seed)
    collocation_points = collect(range(0.0, time_span, length = num_collocation))
    num_layers = length(params)

    adam_state = Vector{Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Float64}}}(undef, num_layers)
    loss_history = Vector{Float64}(undef, num_epochs)

    for layer_index in 1:num_layers
    (weights, biases) = params[layer_index]
    adam_state[layer_index] = (
        zeros(size(weights)),
        zeros(size(weights)),
        zeros(size(biases)),
        zeros(size(biases)))
    end

    for epoch in 1:num_epochs
        grad_params = compute_gradients_autograd(params, collocation_points, omega, time_span, ic_weight)
        (params, adam_state) = adam_step(params, grad_params, adam_state, learning_rate, epoch)

        current_loss = compute_loss(params, collocation_points, omega, time_span, ic_weight)
        loss_history[epoch] = current_loss
        println("Epoch: ", epoch, " | Loss: ", current_loss)
    end

    return (params, loss_history)
end