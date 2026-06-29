using LinearAlgebra
using Statistics
using Random

function init_params(layer_sizes, random_seed)
    Random.seed!(random_seed)
    num_layers = length(layer_sizes) - 1
    params = Vector{Tuple{Matrix{Float64}, Vector{Float64}}}(undef, num_layers)

    for layer_index in 1:num_layers
        fan_in = layer_sizes[layer_index]
        fan_out = layer_sizes[layer_index + 1]
        glorot_scale = sqrt(2.0 / (fan_in + fan_out))
        weights = glorot_scale * randn(fan_out, fan_in)
        biases = zeros(fan_out)
        params[layer_index] = (weights, biases)
    end

    return params
end

function forward(time_input, params)
    num_layers = length(params)
    activation = time_input
    cache = Vector{NamedTuple{(:tanh_output, :tanh_grad), Tuple{Vector{Float64}, Vector{Float64}}}}(undef, num_layers - 1)

    for layer_index in 1:num_layers - 1
        (weights, biases) = params[layer_index]
        pre_activation = weights * activation + biases
        tanh_output = tanh.(pre_activation)
        tanh_grad = 1.0 .- tanh_output .^ 2
        cache[layer_index] = (tanh_output = tanh_output, tanh_grad = tanh_grad)
        activation = tanh_output
    end

    (output_weights, output_biases) = params[num_layers]
    final_output = output_weights * activation + output_biases

    return (final_output, cache)
end

function u_fn(t, params)
    final_output, cache = forward([t], params)
    return first(final_output)
end

function du_fn(t, params)
    num_layers = length(params)
    _, cache = forward([t], params)

    sensitivity = [1.0]
    
    (output_weights, _) = params[num_layers]
    sensitivity = output_weights' * sensitivity

    for hidden_layer in (num_layers - 1):-1:1
        sensitivity = sensitivity .* cache[hidden_layer].tanh_grad
        (weights, _) = params[hidden_layer]
        sensitivity = weights' * sensitivity
    end

    return first(sensitivity)
end

function ddu_fn(t, params)
    num_layers = length(params)
    _, cache = forward([t], params)

    first_deriv = [1.0]
    second_deriv = [0.0]

    (output_weights, _) = params[num_layers]

    first_deriv = output_weights' * first_deriv
    second_deriv = output_weights' * second_deriv

    for layer_index in (num_layers-1):-1:1
        tanh_second = -2.0 .* cache[layer_index].tanh_output .* cache[layer_index].tanh_grad
        new_second_deriv = second_deriv .* cache[layer_index].tanh_grad .+ first_deriv .* tanh_second

        first_deriv = first_deriv .* cache[layer_index].tanh_grad
        second_deriv = new_second_deriv

        (weights, _) = params[layer_index]

        first_deriv = weights' * first_deriv
        second_deriv = weights' * second_deriv
    end
    
    return first(second_deriv)
end