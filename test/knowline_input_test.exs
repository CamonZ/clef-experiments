defmodule Knowline.InputTest do
  use ExUnit.Case, async: false

  setup_all do
    {:ok, tokenizer} =
      Tokenizers.Tokenizer.from_file("models/knowline-4b-gen5.5-mlx-q5/tokenizer.json")

    %{tokenizer: tokenizer}
  end

  test "matches publisher's prompt bytes, token ids, option order and read-twice boundary", %{
    tokenizer: tokenizer
  } do
    fixture = File.read!("fixtures/knowline_prompts.json") |> Jason.decode!()

    ordered =
      File.read!("fixtures/knowline_prompts.json") |> Jason.decode!(objects: :ordered_objects)

    for {entry, expected} <- Enum.zip(ordered["records"], fixture["records"]) do
      result = Knowline.Input.encode(tokenizer, entry["record"])
      assert result.twice == expected["twice"]
      assert result.once_tokens == expected["once_tokens"]

      for {row, ref} <- Enum.zip(result.rows, expected["rows"]) do
        assert row.id == ref["id"]
        assert row.prompt == ref["prompt"]
        assert row.ids == ref["ids"]
        assert row.keys == ref["keys"]
        assert row.label_ids == ref["label_ids"]
      end

      cutoff = result.once_tokens
      assert Knowline.Input.encode(tokenizer, entry["record"], twice_max_tokens: cutoff).twice
      refute Knowline.Input.encode(tokenizer, entry["record"], twice_max_tokens: cutoff - 1).twice

      assert_raise ArgumentError, ~r/exceeds input token limit/, fn ->
        Knowline.Input.encode(tokenizer, entry["record"], max_length: 10)
      end
    end
  end

  test "rejects invalid schemas and unsupported media", %{tokenizer: tokenizer} do
    for json <- [
          "null",
          "[]",
          "{}",
          ~s({"state":"x","questions":[]}),
          ~s({"images":[],"questions":{"q":{"type":"noul"}}}),
          ~s({"state":"x","questions":{"q":{"type":"choice","criteria":{"only":"one"}}}})
        ] do
      assert_raise ArgumentError, fn ->
        Knowline.Input.encode(tokenizer, Knowline.Input.decode!(json))
      end
    end
  end

  test "raw EMLX supports five-bit packed matmul without Nx s5 tensors" do
    w = Nx.iota({32, 64}, type: :f32) |> Nx.divide(100)
    {q, s, b} = EMLX.quantize(EMLX.Backend.from_nx(w), 64, 5)
    x = Nx.broadcast(1.0, {1, 64})

    actual =
      EMLX.quantized_matmul(EMLX.Backend.from_nx(x), q, s, b, true, 64, 5) |> EMLX.Backend.to_nx()

    dense = EMLX.dequantize(q, s, b, 64, 5) |> EMLX.Backend.to_nx()
    expected = Nx.dot(x, [1], dense, [1])
    assert Nx.shape(actual) == {1, 32}
    assert Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(actual, expected)))) < 0.001
  end
end
