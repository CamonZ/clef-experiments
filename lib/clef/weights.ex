defmodule Clef.Weights do
  @moduledoc false

  def load(path, opts \\ []) do
    config = path |> Path.join("config.json") |> File.read!() |> Jason.decode!()

    unless config["model_type"] == "qwen3_5" and
             config["quantization"]["bits"] in Keyword.get(opts, :bits, [8]) and
             config["quantization"]["group_size"] == 64 and
             config["quantization"]["mode"] == "affine" do
      raise ArgumentError,
            "expected an affine group-64 Qwen3.5 MLX checkpoint with supported bit width"
    end

    shards = Path.wildcard(Path.join(path, "model*.safetensors"))
    bits = config["quantization"]["bits"]

    if shards == [] do
      raise ArgumentError, "no model shards found in #{path}"
    end

    raw =
      Enum.reduce(shards, %{}, fn shard, tensors ->
        Map.merge(tensors, Clef.Safetensors.read!(shard))
      end)

    quantized =
      raw
      |> Enum.filter(fn {key, _} -> String.ends_with?(key, ".scales") end)
      |> Map.new(fn {key, _} ->
        name = String.replace_suffix(key, ".scales", "")
        {name, quantized(raw, name, bits)}
      end)

    %{raw: raw, quantized: quantized, config: config["text_config"], bits: bits}
  end

  def fetch(weights, name), do: Map.fetch!(weights.raw, name)

  def linear(weights, name, input) do
    %{weight: weight, scales: scales, biases: biases, bits: bits} =
      Map.fetch!(weights.quantized, name)

    EMLX.quantized_matmul(EMLX.Backend.from_nx(input), weight, scales, biases, true, 64, bits)
    |> EMLX.Backend.to_nx()
  end

  def linear_rows(weights, name, input, ids) do
    packed = gathered(weights, name, ids)

    EMLX.quantized_matmul(
      EMLX.Backend.from_nx(input),
      packed.weight,
      packed.scales,
      packed.biases,
      true,
      64,
      packed.bits
    )
    |> EMLX.Backend.to_nx()
  end

  def rows(weights, name, ids) do
    %{weight: weight, scales: scales, biases: biases, bits: bits} = gathered(weights, name, ids)

    EMLX.dequantize(weight, scales, biases, 64, bits) |> EMLX.Backend.to_nx()
  end

  defp gathered(weights, name, ids) do
    indices = Nx.tensor(ids, type: :s32)

    raw =
      for suffix <- ["weight", "scales", "biases"], into: %{} do
        key = name <> "." <> suffix
        {key, Nx.take(Map.fetch!(weights.raw, key), indices)}
      end

    quantized(raw, name, weights.bits)
  end

  defp quantized(raw, name, bits) do
    %{
      weight: EMLX.Backend.from_nx(Map.fetch!(raw, name <> ".weight")),
      scales: EMLX.Backend.from_nx(Map.fetch!(raw, name <> ".scales")),
      biases: EMLX.Backend.from_nx(Map.fetch!(raw, name <> ".biases")),
      bits: bits
    }
  end
end
