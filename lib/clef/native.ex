defmodule Clef.Native do
  @moduledoc false

  def load_plugin! do
    path = Application.app_dir(:clef, "priv/libclef.so")
    EMLX.NIF.load_plugin("clef", path) |> EMLX.unwrap!()
    :ok
  end

  def pointwise(name, tensors)
      when name in ["silu", "swiglu", "decay"] do
    operands = Enum.map(tensors, &EMLX.Backend.from_nx/1)
    [{device, _} | _] = operands
    {worker, effective_device} = EMLX.resolve_worker(device)

    [reference] =
      EMLX.NIF.call_plugin(
        worker,
        "clef",
        name,
        Enum.map(operands, &elem(&1, 1)),
        [],
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    EMLX.Backend.to_nx({effective_device, reference})
  end

  def gated_delta(queries, keys, values, decay, update_gate) do
    operands = Enum.map([queries, keys, values, decay, update_gate], &EMLX.Backend.from_nx/1)
    [{device, _} | _] = operands
    {worker, effective_device} = EMLX.resolve_worker(device)

    [reference] =
      EMLX.NIF.call_plugin(
        worker,
        "clef",
        "gated_delta",
        Enum.map(operands, &elem(&1, 1)),
        [],
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    EMLX.Backend.to_nx({effective_device, reference}, Nx.to_template(values))
  end
end
