defmodule Clef.Math do
  @moduledoc false

  def silu(input) do
    ones = Nx.broadcast(Nx.tensor(1.0, type: Nx.type(input)), Nx.shape(input))
    EMLX.Fast.swiglu(input, ones)
  end

  def softmax(input, opts \\ [axis: -1]) do
    axis = Keyword.fetch!(opts, :axis)
    maximum = Nx.reduce_max(input, axes: [axis], keep_axes: true)
    exponentials = input |> Nx.subtract(maximum) |> Nx.exp()
    Nx.divide(exponentials, Nx.sum(exponentials, axes: [axis], keep_axes: true))
  end

  def gelu(input) do
    half_input = Nx.multiply(input, 0.5)
    gaussian = input |> Nx.divide(:math.sqrt(2.0)) |> Nx.erf()

    Nx.multiply(half_input, Nx.add(1.0, gaussian))
  end

  def slice(input, axis, start, length) do
    Nx.slice_along_axis(input, start, length, axis: axis)
  end

  def normalize(input) do
    norm =
      input
      |> Nx.pow(2)
      |> Nx.sum(axes: [-1], keep_axes: true)
      |> Nx.sqrt()
      |> Nx.max(1.0e-12)

    Nx.divide(input, norm)
  end
end
