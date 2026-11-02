defmodule Clef.Head do
  @moduledoc "Clef's joint schema head, evaluated in float32."

  alias Clef.{Math, Weights}

  def load(path) do
    weights =
      path
      |> Path.join("joint_head.safetensors")
      |> Clef.Safetensors.read!()
      |> Map.new(fn {name, tensor} -> {name, Nx.as_type(tensor, :f32)} end)

    config = path |> Path.join("joint_head_config.json") |> File.read!() |> Jason.decode!()
    %{weights: weights, config: config}
  end

  def forward(head, backbone, hidden, ids, questions) do
    weights = head.weights
    hidden = hidden |> Nx.as_type(:f32) |> layer_norm(weights, "hidden_norm")
    memory = linear(hidden, weights, "memory_projection", false)
    global_context = row(hidden, elem(Nx.shape(hidden), 0) - 1)
    question_vectors = Enum.map(questions, &mean_span(hidden, &1.qspan)) |> Nx.stack()

    option_contexts =
      Enum.map(questions, fn question ->
        question.ospans |> Enum.map(&mean_span(hidden, &1)) |> Nx.stack()
      end)

    lexical_options = lexical_options(backbone, ids, questions)

    options =
      weights
      |> option_queries(question_vectors, option_contexts, lexical_options)
      |> route_evidence(head, memory)
      |> split_options(questions)

    question_base = linear(question_vectors, weights, "question_projection", false)
    summary = summarize_options(options, question_base, head.config["width"])
    types = Nx.tensor(Enum.map(questions, & &1.type), type: :s32)

    fields =
      question_base
      |> Nx.add(layer_norm(summary, weights, "option_summary_norm"))
      |> Nx.add(linear(global_context, weights, "global_projection", false))
      |> Nx.add(Nx.take(weights["type_embedding.weight"], types))
      |> refine_fields(head, memory)

    score_options(weights, options, lexical_options, question_vectors, global_context, fields)
  end

  defp lexical_options(backbone, ids, questions) do
    Enum.map(questions, fn question ->
      question.ospans
      |> Enum.map(fn {start, stop} ->
        option_ids = Enum.slice(ids, start, stop - start)

        backbone
        |> Weights.rows("language_model.lm_head", option_ids)
        |> Nx.as_type(:f32)
        |> Nx.mean(axes: [0])
      end)
      |> Nx.stack()
    end)
  end

  defp option_queries(weights, question_vectors, contexts, lexical_options) do
    Enum.zip(contexts, lexical_options)
    |> Enum.with_index()
    |> Enum.map(fn {{context, lexical}, index} ->
      context
      |> linear(weights, "option_context_projection", false)
      |> Nx.add(linear(lexical, weights, "option_lexical_projection", false))
      |> Nx.add(
        linear(row(question_vectors, index), weights, "option_question_projection", false)
      )
    end)
    |> Nx.concatenate()
  end

  # Each option query gathers evidence from the complete input memory.
  defp route_evidence(queries, head, memory) do
    weights = head.weights
    heads = head.config["heads"]

    Enum.reduce(0..(head.config["routing_layers"] - 1), queries, fn index, queries ->
      prefix = "evidence_layers.#{index}"
      query_input = layer_norm(queries, weights, prefix <> ".query_norm")
      memory_input = layer_norm(memory, weights, prefix <> ".memory_norm")

      attention =
        multihead_attention(weights, prefix <> ".attention", query_input, memory_input, heads)

      queries = Nx.add(queries, attention)

      feedforward =
        queries
        |> layer_norm(weights, prefix <> ".feedforward_norm")
        |> linear(weights, prefix <> ".feedforward.0")
        |> Math.gelu()
        |> linear(weights, prefix <> ".feedforward.3")

      Nx.add(queries, feedforward)
    end)
  end

  # Restore per-question groups after routing all options in one tensor.
  defp split_options(queries, questions) do
    {options, _offset} =
      Enum.map_reduce(questions, 0, fn question, offset ->
        count = length(question.ospans)
        {Math.slice(queries, 0, offset, count), offset + count}
      end)

    options
  end

  defp summarize_options(options, question_base, width) do
    options
    |> Enum.with_index()
    |> Enum.map(fn {question_options, index} ->
      probabilities =
        question_options
        |> Nx.dot(row(question_base, index))
        |> Nx.divide(:math.sqrt(width))
        |> Math.softmax(axis: 0)
        |> Nx.new_axis(-1)

      Nx.multiply(probabilities, question_options) |> Nx.sum(axes: [0])
    end)
    |> Nx.stack()
  end

  # Field self-attention lets questions inform each other before scoring.
  defp refine_fields(fields, head, memory) do
    weights = head.weights
    heads = head.config["heads"]

    Enum.reduce(0..(head.config["layers"] - 1), fields, fn index, fields ->
      prefix = "layers.#{index}"
      self_input = layer_norm(fields, weights, prefix <> ".norm1")

      self_attention =
        multihead_attention(weights, prefix <> ".self_attn", self_input, self_input, heads)

      fields = Nx.add(fields, self_attention)

      query_input = layer_norm(fields, weights, prefix <> ".norm2")

      memory_attention =
        multihead_attention(weights, prefix <> ".multihead_attn", query_input, memory, heads)

      fields = Nx.add(fields, memory_attention)

      feedforward =
        fields
        |> layer_norm(weights, prefix <> ".norm3")
        |> linear(weights, prefix <> ".linear1")
        |> Math.gelu()
        |> linear(weights, prefix <> ".linear2")

      Nx.add(fields, feedforward)
    end)
    |> layer_norm(weights, "field_norm")
  end

  defp score_options(weights, options, lexical_options, question_vectors, global_context, fields) do
    prior_scale = Nx.exp(Nx.min(weights["prior_logit_scale"], :math.log(100.0)))
    joint_scale = Nx.exp(Nx.min(weights["joint_logit_scale"], :math.log(100.0)))
    residual_gate = Nx.sigmoid(weights["residual_gate"])

    options
    |> Enum.with_index()
    |> Enum.map(fn {question_options, index} ->
      anchor = Math.normalize(Nx.add(row(question_vectors, index), global_context))

      prior =
        lexical_options
        |> Enum.at(index)
        |> Math.normalize()
        |> Nx.dot(anchor)
        |> Nx.multiply(prior_scale)

      normalized_options = layer_norm(question_options, weights, "option_norm")
      field = Nx.broadcast(row(fields, index), Nx.shape(normalized_options))

      cosine_similarity =
        Nx.sum(Nx.multiply(field, normalized_options), axes: [-1])
        |> Nx.divide(Nx.max(Nx.multiply(l2_norm(field), l2_norm(normalized_options)), 1.0e-8))

      features =
        Nx.concatenate(
          [
            field,
            normalized_options,
            Nx.multiply(field, normalized_options),
            Nx.abs(Nx.subtract(field, normalized_options))
          ],
          axis: -1
        )

      residual =
        features
        |> linear(weights, "residual_scorer.0")
        |> Math.gelu()
        |> linear(weights, "residual_scorer.3")
        |> Nx.squeeze(axes: [-1])

      joint_score = Nx.add(Nx.multiply(joint_scale, cosine_similarity), residual)
      Nx.add(prior, Nx.multiply(residual_gate, joint_score))
    end)
  end

  defp row(tensor, index), do: Nx.take(tensor, Nx.tensor(index))

  defp mean_span(tensor, {start, stop}) do
    tensor
    |> Math.slice(0, start, stop - start)
    |> Nx.mean(axes: [0])
  end

  defp l2_norm(tensor) do
    tensor |> Nx.pow(2) |> Nx.sum(axes: [-1]) |> Nx.sqrt()
  end

  defp linear(input, weights, prefix, bias? \\ true) do
    projected = Nx.dot(input, [Nx.rank(input) - 1], Map.fetch!(weights, prefix <> ".weight"), [1])

    if bias? do
      Nx.add(projected, Map.fetch!(weights, prefix <> ".bias"))
    else
      projected
    end
  end

  defp layer_norm(input, weights, prefix) do
    centered = Nx.subtract(input, Nx.mean(input, axes: [-1], keep_axes: true))
    variance = Nx.pow(centered, 2) |> Nx.mean(axes: [-1], keep_axes: true)

    centered
    |> Nx.multiply(Nx.rsqrt(Nx.add(variance, 1.0e-5)))
    |> Nx.multiply(Map.fetch!(weights, prefix <> ".weight"))
    |> Nx.add(Map.fetch!(weights, prefix <> ".bias"))
  end

  defp multihead_attention(weights, prefix, query_input, key_value_input, heads) do
    {query_count, width} = Nx.shape(query_input)
    head_dimension = div(width, heads)
    queries = attention_projection(query_input, weights, prefix, 0, heads, width)
    keys = attention_projection(key_value_input, weights, prefix, 1, heads, width)
    values = attention_projection(key_value_input, weights, prefix, 2, heads, width)

    EMLX.Fast.scaled_dot_product_attention(
      queries,
      keys,
      values,
      1.0 / :math.sqrt(head_dimension)
    )
    |> Nx.squeeze(axes: [0])
    |> Nx.transpose(axes: [1, 0, 2])
    |> Nx.reshape({query_count, width})
    |> linear(weights, prefix <> ".out_proj")
  end

  defp attention_projection(input, weights, prefix, index, heads, width) do
    token_count = elem(Nx.shape(input), 0)
    head_dimension = div(width, heads)

    # The checkpoint stores query, key and value projections in consecutive rows.
    weight = Math.slice(Map.fetch!(weights, prefix <> ".in_proj_weight"), 0, index * width, width)
    bias = Math.slice(Map.fetch!(weights, prefix <> ".in_proj_bias"), 0, index * width, width)

    Nx.dot(input, [1], weight, [1])
    |> Nx.add(bias)
    |> Nx.reshape({token_count, heads, head_dimension})
    |> Nx.transpose(axes: [1, 0, 2])
    |> Nx.new_axis(0)
  end
end
