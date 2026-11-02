defmodule Clef.Backbone do
  @moduledoc "Qwen3.5 dense text prefill on EMLX. Every request starts with fresh recurrent state."

  alias Clef.{Math, Weights}

  def forward(weights, ids, opts \\ []) do
    embeddings =
      weights
      |> Weights.rows("language_model.model.embed_tokens", ids)
      |> Nx.new_axis(0)
      |> trace("embedding", opts)

    hidden =
      Enum.reduce(0..(weights.config["num_hidden_layers"] - 1), embeddings, fn index, hidden ->
        weights
        |> layer(index, hidden)
        |> trace("layer_#{index}", opts)
      end)

    normalized = rms_norm(hidden, weights, "language_model.model.norm.weight")

    if callback = opts[:trace] do
      callback.("hidden", Nx.squeeze(normalized, axes: [0]))
    end

    normalized
  end

  def layer(weights, index, hidden) do
    prefix = "language_model.model.layers.#{index}"
    attention_input = rms_norm(hidden, weights, prefix <> ".input_layernorm.weight")

    attention_output =
      if rem(index + 1, weights.config["full_attention_interval"]) == 0 do
        full_attention(weights, prefix <> ".self_attn", attention_input)
      else
        gated_delta_attention(weights, prefix <> ".linear_attn", attention_input)
      end

    hidden = Nx.add(hidden, attention_output)
    feedforward_input = rms_norm(hidden, weights, prefix <> ".post_attention_layernorm.weight")
    Nx.add(hidden, feedforward(weights, prefix <> ".mlp", feedforward_input))
  end

  defp trace(tensor, name, opts) do
    if callback = opts[:trace] do
      callback.(name, tensor)
    end

    tensor
  end

  defp rms_norm(input, weights, name) do
    EMLX.Fast.rms_norm(input, Weights.fetch(weights, name), weights.config["rms_norm_eps"] * 1.0)
  end

  defp feedforward(weights, prefix, input) do
    gate = Weights.linear(weights, prefix <> ".gate_proj", input)
    up = Weights.linear(weights, prefix <> ".up_proj", input)

    swiglu(gate, up, weights)
    |> then(&Weights.linear(weights, prefix <> ".down_proj", &1))
  end

  defp full_attention(weights, prefix, input) do
    {batch_size, sequence_length, _} = Nx.shape(input)
    config = weights.config
    query_heads = config["num_attention_heads"]
    key_value_heads = config["num_key_value_heads"]
    head_dimension = config["head_dim"]

    # The query projection packs each query together with its output gate.
    query_and_gate =
      weights
      |> Weights.linear(prefix <> ".q_proj", input)
      |> Nx.reshape({batch_size, sequence_length, query_heads, head_dimension * 2})

    gate =
      query_and_gate
      |> Math.slice(3, head_dimension, head_dimension)
      |> Nx.reshape({batch_size, sequence_length, query_heads * head_dimension})

    queries =
      query_and_gate
      |> Math.slice(3, 0, head_dimension)
      |> rms_norm(weights, prefix <> ".q_norm.weight")

    keys =
      weights
      |> Weights.linear(prefix <> ".k_proj", input)
      |> Nx.reshape({batch_size, sequence_length, key_value_heads, head_dimension})
      |> rms_norm(weights, prefix <> ".k_norm.weight")

    values =
      weights
      |> Weights.linear(prefix <> ".v_proj", input)
      |> Nx.reshape({batch_size, sequence_length, key_value_heads, head_dimension})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    queries = apply_rotary_positions(queries, config)
    keys = apply_rotary_positions(keys, config)

    EMLX.Fast.scaled_dot_product_attention_causal(
      queries,
      keys,
      values,
      :math.pow(head_dimension, -0.5)
    )
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch_size, sequence_length, query_heads * head_dimension})
    |> Nx.multiply(Nx.sigmoid(gate))
    |> then(&Weights.linear(weights, prefix <> ".o_proj", &1))
  end

  defp apply_rotary_positions(input, config) do
    rope = config["rope_parameters"]
    rotary_dimensions = round(config["head_dim"] * rope["partial_rotary_factor"])

    input
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> EMLX.Fast.rope(rotary_dimensions, false, rope["rope_theta"] * 1.0, 1.0, 0)
  end

  defp gated_delta_attention(weights, prefix, input) do
    config = weights.config
    {batch_size, sequence_length, _} = Nx.shape(input)
    key_heads = config["linear_num_key_heads"]
    value_heads = config["linear_num_value_heads"]
    key_head_dimension = config["linear_key_head_dim"]
    value_head_dimension = config["linear_value_head_dim"]
    key_width = key_heads * key_head_dimension
    value_width = value_heads * value_head_dimension
    query_key_value = Weights.linear(weights, prefix <> ".in_proj_qkv", input)

    output_gate =
      weights
      |> Weights.linear(prefix <> ".in_proj_z", input)
      |> Nx.reshape({batch_size, sequence_length, value_heads, value_head_dimension})

    decay_input = Weights.linear(weights, prefix <> ".in_proj_a", input)
    update_input = Weights.linear(weights, prefix <> ".in_proj_b", input)
    kernel = Weights.fetch(weights, prefix <> ".conv1d.weight")

    padding = config["linear_conv_kernel_dim"] - 1

    {convolution_input, convolution_padding} =
      if weights[:compiled_pointwise] do
        # Match mlx-lm's explicit recurrent-state prefix and valid convolution.
        zeros =
          Nx.broadcast(
            Nx.tensor(0, type: Nx.type(query_key_value)),
            {batch_size, padding, 2 * key_width + value_width}
          )

        {Nx.concatenate([zeros, query_key_value], axis: 1), [{0, 0}]}
      else
        {query_key_value, [{padding, 0}]}
      end

    convolved =
      Nx.conv(convolution_input, kernel,
        padding: convolution_padding,
        feature_group_size: 2 * key_width + value_width,
        input_permutation: [0, 2, 1],
        kernel_permutation: [0, 2, 1],
        output_permutation: [0, 2, 1]
      )
      |> silu(weights)

    queries =
      convolved
      |> Math.slice(2, 0, key_width)
      |> Nx.reshape({batch_size, sequence_length, key_heads, key_head_dimension})

    keys =
      convolved
      |> Math.slice(2, key_width, key_width)
      |> Nx.reshape({batch_size, sequence_length, key_heads, key_head_dimension})

    values =
      convolved
      |> Math.slice(2, key_width * 2, value_width)
      |> Nx.reshape({batch_size, sequence_length, value_heads, value_head_dimension})

    # The reference uses unweighted RMS normalization for queries and keys.
    ones = Nx.broadcast(Nx.tensor(1.0, type: Nx.type(queries)), {key_head_dimension})

    queries =
      EMLX.Fast.rms_norm(queries, ones, 1.0e-6)
      |> Nx.multiply(1.0 / key_head_dimension)

    key_scale = 1.0 / :math.sqrt(key_head_dimension)

    key_scale =
      if weights[:compiled_pointwise],
        do: Nx.tensor(key_scale, type: Nx.type(keys)),
        else: key_scale

    keys =
      EMLX.Fast.rms_norm(keys, ones, 1.0e-6)
      |> Nx.multiply(key_scale)

    decay = delta_decay(weights, prefix, decay_input)
    output = Clef.Native.gated_delta(queries, keys, values, decay, Nx.sigmoid(update_input))
    normalized = output |> rms_norm(weights, prefix <> ".norm.weight") |> Nx.as_type(:f32)

    Nx.multiply(normalized, silu(Nx.as_type(output_gate, :f32), weights))
    |> Nx.as_type(Nx.type(output))
    |> Nx.reshape({batch_size, sequence_length, value_width})
    |> then(&Weights.linear(weights, prefix <> ".out_proj", &1))
  end

  defp silu(input, %{compiled_pointwise: true}), do: Clef.Native.pointwise("silu", [input])
  defp silu(input, _weights), do: Math.silu(input)

  defp swiglu(gate, up, %{compiled_pointwise: true}),
    do: Clef.Native.pointwise("swiglu", [gate, up])

  defp swiglu(gate, up, _weights), do: EMLX.Fast.swiglu(gate, up)

  defp delta_decay(%{compiled_pointwise: true} = weights, prefix, input) do
    Clef.Native.pointwise("decay", [
      Weights.fetch(weights, prefix <> ".A_log"),
      input,
      Weights.fetch(weights, prefix <> ".dt_bias")
    ])
  end

  defp delta_decay(weights, prefix, input) do
    # The addition is rounded to bf16 before softplus, as in mlx-lm.
    softplus_input =
      Nx.add(input, Weights.fetch(weights, prefix <> ".dt_bias")) |> Nx.as_type(:f32)

    # This stable softplus form avoids overflow for large positive inputs.
    positive_part = Nx.max(softplus_input, 0.0)
    exponential_tail = softplus_input |> Nx.abs() |> Nx.negate() |> Nx.exp() |> Nx.log1p()
    softplus = Nx.add(positive_part, exponential_tail)

    weights
    |> Weights.fetch(prefix <> ".A_log")
    |> Nx.as_type(:f32)
    |> Nx.exp()
    |> Nx.negate()
    |> Nx.multiply(softplus)
    |> Nx.exp()
  end
end
