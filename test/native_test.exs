defmodule Clef.NativeTest do
  use ExUnit.Case, async: false

  test "Metal recurrence matches independent mlx-lm operations with grouped heads and batches" do
    tensors = Clef.Safetensors.read!("fixtures/delta.safetensors")

    run = fn ->
      Clef.Native.gated_delta(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["g"],
        tensors["beta"]
      )
    end

    actual = run.()

    maximum_error =
      actual
      |> Nx.subtract(tensors["y"])
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert maximum_error < 1.0e-6
    assert Nx.to_binary(run.()) == Nx.to_binary(actual)
  end
end
