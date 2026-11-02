defmodule Knowline do
  @moduledoc "Native EMLX KnowLine Gen5.5 inference with label-token probability readout."
  alias Knowline.Input
  @embedding "language_model.model.embed_tokens"

  def load(path) do
    Clef.Native.load_plugin!()
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_file(Path.join(path, "tokenizer.json"))

    %{
      backbone: Clef.Weights.load(path, bits: [5]) |> Map.put(:compiled_pointwise, true),
      tokenizer: tokenizer,
      name: Path.basename(Path.expand(path)),
      labels: Input.labels(tokenizer)
    }
  end

  def decide(model, record, opts \\ []) do
    encoded = Input.encode(model.tokenizer, record, Keyword.put(opts, :labels, model.labels))

    probabilities =
      Map.new(encoded.rows, fn row ->
        hidden = Clef.Backbone.forward(model.backbone, row.ids, opts)

        last =
          hidden
          |> Nx.slice_along_axis(length(row.ids) - 1, 1, axis: 1)
          |> Nx.reshape({1, model.backbone.config["hidden_size"]})

        # Score only packed label rows, retaining MLX's quantized readout precision.
        logits =
          Clef.Weights.linear_rows(model.backbone, @embedding, last, row.label_ids)
          |> Nx.as_type(:f32)
          |> Nx.reshape({length(row.keys)})

        probs = logits |> Clef.Math.softmax() |> Nx.to_flat_list()
        {row.id, Map.new(Enum.zip(row.keys, probs))}
      end)

    if Keyword.get(opts, :envelope, false) do
      answers =
        Map.new(encoded.rows, fn row ->
          {row.id, answer(row.question, row.keys, probabilities[row.id])}
        end)

      %{
        model: model.name,
        answers: answers,
        usage: %{
          input_tokens: Enum.sum(Enum.map(encoded.rows, &length(&1.ids))),
          output_tokens: length(encoded.rows)
        }
      }
    else
      probabilities
    end
  end

  defp answer(q, keys, probabilities) do
    confidence =
      1.0 +
        Enum.reduce(probabilities, 0.0, fn {_, p}, acc ->
          if p > 0, do: acc + p * :math.log(p), else: acc
        end) / :math.log(length(keys))

    case Input.get(q, "type") do
      "noul" ->
        %{type: "noul", noul: probabilities["true"]}

      "choice" ->
        %{
          type: "choice",
          choice: Enum.max_by(keys, &probabilities[&1]),
          probabilities: probabilities,
          confidence: max(0.0, min(1.0, confidence))
        }

      "score" ->
        %{
          type: "score",
          score:
            Enum.with_index(keys)
            |> Enum.reduce(0.0, fn {key, i}, sum -> sum + i * probabilities[key] end),
          probabilities: probabilities,
          legend:
            Input.get(q, "criteria")
            |> Enum.with_index()
            |> Map.new(fn {v, i} -> {Integer.to_string(i), v} end),
          confidence: max(0.0, min(1.0, confidence))
        }
    end
  end
end
