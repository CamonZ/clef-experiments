defmodule Clef.Safetensors do
  @moduledoc "Reads checkpoint tensors incrementally without holding a whole shard in BEAM memory."

  @types %{
    "BF16" => :bf16,
    "F16" => :f16,
    "F32" => :f32,
    "I32" => :s32,
    "I64" => :s64,
    "U32" => :u32
  }

  def read!(path, opts \\ []) do
    {:ok, file} = :file.open(String.to_charlist(path), [:read, :binary, :raw])

    try do
      {:ok, <<header_size::unsigned-little-64>>} = :file.pread(file, 0, 8)
      {:ok, json} = :file.pread(file, 8, header_size)
      header = Jason.decode!(json) |> Map.delete("__metadata__")

      header
      |> Enum.sort_by(fn {_, specification} -> hd(specification["data_offsets"]) end)
      |> Map.new(fn {name, specification} ->
        [start, stop] = specification["data_offsets"]
        {:ok, bytes} = :file.pread(file, 8 + header_size + start, stop - start)

        tensor =
          bytes
          |> Nx.from_binary(Map.fetch!(@types, specification["dtype"]), opts)
          |> Nx.reshape(List.to_tuple(specification["shape"]))

        {name, tensor}
      end)
    after
      :file.close(file)
    end
  end
end
