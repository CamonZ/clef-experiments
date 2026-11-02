defmodule Clef do
  @moduledoc "Native Elixir/Nx inference for the text-only TrevorJS Clef-Flash 8-bit checkpoint."

  def load(path \\ "models/clef-flash-8bit") do
    Clef.Native.load_plugin!()
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_file(Path.join(path, "tokenizer.json"))

    %{
      backbone: Clef.Weights.load(path),
      head: Clef.Head.load(path),
      tokenizer: tokenizer,
      name: Path.basename(Path.expand(path))
    }
  end

  def decide(model, record, opts \\ []) do
    %{ids: ids, fields: fields} =
      Clef.Input.encode(model.tokenizer, record, Keyword.get(opts, :max_length, 16_384))

    hidden =
      model.backbone
      |> Clef.Backbone.forward(ids, opts)
      |> Nx.squeeze(axes: [0])

    logits = Clef.Head.forward(model.head, model.backbone, hidden, ids, fields)

    probabilities =
      Enum.zip(fields, logits)
      |> Map.new(fn {field, field_logits} ->
        probabilities = field_logits |> Clef.Math.softmax() |> Nx.to_flat_list()
        options = field.oids |> Enum.zip(probabilities) |> Map.new()
        {field.id, options}
      end)

    if Keyword.get(opts, :envelope, false) do
      Clef.Response.build(model.name, record, probabilities, length(ids))
    else
      probabilities
    end
  end
end
