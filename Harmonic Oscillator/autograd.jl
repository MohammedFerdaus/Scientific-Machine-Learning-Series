using LinearAlgebra
using Statistics

mutable struct TapeEntry
    output_value::Float64
    input_indices::Vector{Int}
    backward_fn::Function
end

mutable struct Tape
    entries::Vector{TapeEntry}
    gradients::Vector{Float64}
end

mutable struct TrackedValue
    value::Float64
    tape_index::Int
end

function new_tape()
    return Tape(TapeEntry[], Float64[])
end

function track_constant(tape::Tape, value::Float64)
    noop_backward = (upstream_gradient) -> nothing
    entry = TapeEntry(value, Int[], noop_backward)
    
    push!(tape.entries, entry)
    push!(tape.gradients, 0.0)
    
    return TrackedValue(value, length(tape.entries))
end

function track_param(tape::Tape, value::Float64)
    noop_backward = (upstream_gradient) -> nothing
    entry = TapeEntry(value, Int[], noop_backward)
    
    push!(tape.entries, entry)
    push!(tape.gradients, 0.0) 

    return TrackedValue(value, length(tape.entries))
end

function tracked_add(tape, tracked_a, tracked_b)
    output_value = tracked_a.value + tracked_b.value
    backward_fn = (upstream) -> begin
        tape.gradients[tracked_a.tape_index] += upstream * 1.0
        tape.gradients[tracked_b.tape_index] += upstream * 1.0
    end

    entry = TapeEntry(output_value, [tracked_a.tape_index, tracked_b.tape_index], backward_fn)
    
    push!(tape.entries, entry)
    push!(tape.gradients, 0.0)

    return TrackedValue(output_value, length(tape.entries))
end

function tracked_subtract(tape, tracked_a, tracked_b)
    output_value = tracked_a.value - tracked_b.value
    backward_fn = (upstream) -> begin
        tape.gradients[tracked_a.tape_index] += upstream
        tape.gradients[tracked_b.tape_index] += upstream * -1.0 
    end

    entry = TapeEntry(output_value, [tracked_a.tape_index, tracked_b.tape_index], backward_fn)
    
    push!(tape.entries, entry)
    push!(tape.gradients, 0.0)

    return TrackedValue(output_value, length(tape.entries))
end

function tracked_multiply(tape, tracked_a, tracked_b)
    output_value = tracked_a.value * tracked_b.value
    backward_fn = (upstream) -> begin 
        tape.gradients[tracked_a.tape_index] += upstream * tracked_b.value
        tape.gradients[tracked_b.tape_index] += upstream * tracked_a.value
    end

    entry = TapeEntry(output_value, [tracked_a.tape_index, tracked_b.tape_index], backward_fn)
    
    push!(tape.entries, entry)
    push!(tape.gradients, 0.0)

    return TrackedValue(output_value, length(tape.entries))
end

function tracked_tanh(tape, tracked_a)
    output_value = tanh(tracked_a.value)
    local_grad = 1.0 - output_value^2
    backward_fn = (upstream) -> begin
        tape.gradients[tracked_a.tape_index] += upstream * local_grad
    end
    
    entry = TapeEntry(output_value, [tracked_a.tape_index], backward_fn)
    push!(tape.entries, entry)
    push!(tape.gradients, 0.0)

    return TrackedValue(output_value, length(tape.entries))
end

function tracked_matmul(tape, tracked_weight_row, tracked_activation)
    output_value = sum(tracked_weight_row[j].value * tracked_activation[j].value for j in 1:length(tracked_weight_row))
    
    weight_indices = [w.tape_index for w in tracked_weight_row]
    activation_indices = [a.tape_index for a in tracked_activation]
    
    backward_fn = (upstream) -> begin
        for j in 1:length(tracked_weight_row)
            tape.gradients[weight_indices[j]] += upstream * tracked_activation[j].value
            tape.gradients[activation_indices[j]] += upstream * tracked_weight_row[j].value
        end
    end
    
    all_input_indices = [weight_indices; activation_indices]
    entry = TapeEntry(output_value, all_input_indices, backward_fn)

    push!(tape.entries, entry)
    push!(tape.gradients, 0.0)
    
    return TrackedValue(output_value, length(tape.entries))
end

function tracked_mean(tape, tracked_values)
    n = length(tracked_values)
    
    output_value = sum(tracked_values[i].value for i in 1:n) / n
    input_indices = [v.tape_index for v in tracked_values]
    
    backward_fn = (upstream) -> begin
        for i in 1:length(input_indices)
            tape.gradients[input_indices[i]] += upstream / n
        end
    end
    
    entry = TapeEntry(output_value, input_indices, backward_fn)

    push!(tape.entries, entry)
    push!(tape.gradients, 0.0)

    return TrackedValue(output_value, length(tape.entries))
end

function tracked_square(tape, tracked_a)
    output_value = tracked_a.value ^ 2
    
    input_value = tracked_a.value
    backward_fn = (upstream) -> begin
        tape.gradients[tracked_a.tape_index] += upstream * 2.0 * input_value
    end

    entry = TapeEntry(output_value, [tracked_a.tape_index], backward_fn)

    push!(tape.entries, entry)
    push!(tape.gradients, 0.0)

    return TrackedValue(output_value, length(tape.entries))
end

function tracked_negate(tape, tracked_a)
    output_value = -tracked_a.value
    backward_fn = (upstream) -> begin
        tape.gradients[tracked_a.tape_index] += upstream * -1.0
    end

    entry = TapeEntry(output_value, [tracked_a.tape_index], backward_fn)

    push!(tape.entries, entry)
    push!(tape.gradients, 0.0)

    return TrackedValue(output_value, length(tape.entries))
end

function tracked_scale(tape, tracked_a, scalar)
    output_value = tracked_a.value * scalar
    backward_fn = (upstream) -> begin
        tape.gradients[tracked_a.tape_index] += upstream * scalar
    end

    entry = TapeEntry(output_value, [tracked_a.tape_index], backward_fn)

    push!(tape.entries, entry)
    push!(tape.gradients, 0.0)

    return TrackedValue(output_value, length(tape.entries))
end

function backward!(tape, loss_tracked)
    tape.gradients[loss_tracked.tape_index] = 1.0
    for index in length(tape.entries):-1:1
        upstream = tape.gradients[index]
        if upstream != 0.0
            tape.entries[index].backward_fn(upstream)
        end
    end
end



function extract_grad(tape, tracked_value)
    return tape.gradients[tracked_value.tape_index]
end