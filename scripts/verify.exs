records = File.read!("fixtures/records.json") |> Jason.decode!(objects: :ordered_objects)
reference = File.read!("artifacts/reference.json") |> Jason.decode!()
{load_microseconds, model} = :timer.tc(fn -> Clef.load() end)
IO.puts("Native checkpoint loaded in #{Float.round(load_microseconds / 1_000_000, 2)}s")

Enum.zip(records, reference)
|> Enum.with_index()
|> Enum.each(fn {{record, expected}, index} ->
  record = record |> Jason.encode!() |> Clef.Input.decode!()
  encoded = Clef.Input.encode(model.tokenizer, record)

  unless encoded.ids == expected["ids"] do
    raise "token IDs differ for record #{index}"
  end

  for {actual, expected_field} <- Enum.zip(encoded.fields, expected["fields"]) do
    unless Tuple.to_list(actual.qspan) == expected_field["qspan"] and
             Enum.map(actual.ospans, &Tuple.to_list/1) == expected_field["ospans"] and
             actual.oids == expected_field["oids"] and actual.type == expected_field["type"] do
      raise "field metadata differs"
    end
  end

  traces = Clef.Safetensors.read!("artifacts/reference_#{index}.safetensors")

  # Isolate head accuracy by supplying the Python backbone's hidden states.
  head_probabilities =
    Clef.Head.forward(model.head, model.backbone, traces["hidden"], encoded.ids, encoded.fields)
    |> Enum.map(fn logits -> logits |> Clef.Math.softmax() |> Nx.to_flat_list() end)

  head_difference =
    Enum.zip(encoded.fields, head_probabilities)
    |> Enum.flat_map(fn {field, probabilities} ->
      Enum.zip(field.oids, probabilities)
      |> Enum.map(fn {option_id, probability} ->
        abs(probability - expected["probabilities"][field.id][option_id])
      end)
    end)
    |> Enum.max()

  IO.puts(
    "record #{index} head on identical hidden states: max probability diff #{head_difference}"
  )

  unless head_difference < 1.0e-5 do
    raise "joint head differs on identical inputs"
  end

  # Compare the native backbone layer by layer against the Python trace.
  trace = fn name, actual ->
    if expected_tensor = traces[name] do
      difference = Nx.subtract(Nx.as_type(actual, :f32), Nx.as_type(expected_tensor, :f32))
      maximum_error = difference |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      rms_error = difference |> Nx.pow(2) |> Nx.mean() |> Nx.sqrt() |> Nx.to_number()
      IO.puts("record #{index} #{name}: max=#{maximum_error}, rms=#{rms_error}")
    end
  end

  {inference_microseconds, result} = :timer.tc(fn -> Clef.decide(model, record, trace: trace) end)

  maximum_difference =
    Enum.flat_map(result, fn {id, options} ->
      Enum.map(options, fn {option_id, probability} ->
        abs(probability - expected["probabilities"][id][option_id])
      end)
    end)
    |> Enum.max()

  IO.puts(
    "record #{index}: #{Float.round(inference_microseconds / 1_000_000, 3)}s; probability max diff #{maximum_difference}; #{inspect(result)}"
  )

  unless maximum_difference < 0.005 do
    raise "probabilities drifted beyond tolerance"
  end
end)
