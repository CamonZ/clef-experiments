defmodule Clef.Input do
  @moduledoc """
  Clef's text/JSON record encoding. An ordered question list preserves caller field order.

  Field metadata uses `qspan` for the instruction span, `ospans` for option spans,
  and `oids` for option IDs. Spans are `{start, stop}` token offsets with an
  exclusive stop, matching the reference implementation.
  """

  @prompt "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options."
  @types %{"noul" => 0, "choice" => 1, "score" => 2}

  def decode!(json) do
    ordered = Jason.decode!(json, objects: :ordered_objects)

    unless match?(%Jason.OrderedObject{}, ordered) and
             match?(%Jason.OrderedObject{}, ordered["questions"]) do
      raise ArgumentError, "record and questions must be JSON objects"
    end

    record = plain(ordered)

    %{
      record
      | "questions" =>
          Enum.map(ordered["questions"].values, fn {id, question} -> {id, plain(question)} end)
    }
  end

  def encode(tokenizer, record, max_length \\ 16_384) do
    unless is_integer(max_length) and max_length > 0 do
      raise ArgumentError, "max_length must be a positive integer"
    end

    state = Map.fetch!(record, "state")
    questions = record |> Map.fetch!("questions") |> question_pairs()

    if questions == [] do
      raise ArgumentError, "at least one question is required"
    end

    if Map.has_key?(record, "images") or Map.has_key?(record, "videos") do
      raise ArgumentError, "this checkpoint supports text/JSON only"
    end

    {schema_ids, schema_fields} = encode_schema(tokenizer, questions)

    prefix_ids =
      tokens(tokenizer, "<|im_start|>system\n#{@prompt}<|im_end|>\n<|im_start|>user\nSTATE:\n")

    suffix_ids =
      tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
      )

    # Reserve the complete schema and chat framing before truncating the state.
    state_budget = max_length - length(prefix_ids) - length(schema_ids) - length(suffix_ids)

    if state_budget < 0 do
      raise ArgumentError, "schema exceeds the input token limit"
    end

    state_ids = tokenizer |> tokens(render(state)) |> Enum.take(state_budget)
    schema_offset = length(prefix_ids) + length(state_ids)

    fields =
      Enum.map(schema_fields, fn field ->
        %{
          field
          | qspan: shift_span(field.qspan, schema_offset),
            ospans: Enum.map(field.ospans, &shift_span(&1, schema_offset))
        }
      end)

    %{ids: prefix_ids ++ state_ids ++ schema_ids ++ suffix_ids, fields: fields}
  end

  def tokens(tokenizer, text) do
    {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, text, add_special_tokens: false)
    Tokenizers.Encoding.get_ids(encoding)
  end

  def render(value) when is_binary(value), do: value
  def render(value), do: value |> canonical() |> Jason.encode!()

  defp encode_schema(tokenizer, questions) do
    schema_prefix = tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")

    questions
    |> Enum.with_index(1)
    |> Enum.reduce({schema_prefix, []}, fn {{id, question}, index}, {ids, fields} ->
      {ids, field} = encode_field(tokenizer, id, question, index, ids)
      {ids, fields ++ [field]}
    end)
  end

  defp encode_field(tokenizer, id, question, index, ids) do
    unless is_binary(id) and is_map(question) do
      raise ArgumentError, "question IDs must be strings and questions must be objects"
    end

    type = Map.fetch!(question, "type")

    unless Map.has_key?(@types, type) do
      raise ArgumentError, "unknown question type #{inspect(type)}"
    end

    ids = ids ++ tokens(tokenizer, "\nFIELD #{index}\nID: #{id}\nTYPE: #{type}\nINSTRUCTION: ")
    instruction_start = length(ids)
    instruction = question["instructions"]

    instruction =
      if instruction in [nil, "", false, 0, 0.0, [], %{}] do
        id
      else
        instruction
      end

    ids = ids ++ tokens(tokenizer, render(instruction))
    question_span = {instruction_start, length(ids)}
    ids = ids ++ tokens(tokenizer, "\nALLOWED OPTIONS:\n")
    options = options(question)
    {ids, option_spans} = encode_options(tokenizer, options, ids)

    field = %{
      id: id,
      type: @types[type],
      qspan: question_span,
      ospans: option_spans,
      oids: Enum.map(options, &elem(&1, 0))
    }

    {ids ++ tokens(tokenizer, "END FIELD\n"), field}
  end

  defp encode_options(tokenizer, options, ids) do
    options
    |> Enum.with_index(1)
    |> Enum.reduce({ids, []}, fn {{option_id, description}, number}, {ids, spans} ->
      ids = ids ++ tokens(tokenizer, "OPTION #{number}: ")
      option_start = length(ids)

      value =
        if is_nil(description) do
          %{"option_id" => option_id}
        else
          %{"option_id" => option_id, "description" => description}
        end

      ids = ids ++ tokens(tokenizer, render(value))
      option_span = {option_start, length(ids)}
      {ids ++ tokens(tokenizer, "\n"), spans ++ [option_span]}
    end)
  end

  defp plain(%Jason.OrderedObject{values: pairs}) do
    Map.new(pairs, fn {key, value} -> {key, plain(value)} end)
  end

  defp plain(list) when is_list(list), do: Enum.map(list, &plain/1)
  defp plain(value), do: value

  # Python json.dumps(sort_keys=True, ensure_ascii=False, separators=(",", ":")).
  defp canonical(%Jason.OrderedObject{values: pairs}), do: canonical(Map.new(pairs))

  defp canonical(value) when is_map(value) do
    value
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.map(fn {key, value} -> {key, canonical(value)} end)
    |> Jason.OrderedObject.new()
  end

  defp canonical(value) when is_list(value), do: Enum.map(value, &canonical/1)
  defp canonical(value) when is_float(value), do: Jason.Fragment.new(python_float(value))
  defp canonical(value), do: value

  # JSON numbers use Python's fixed/scientific thresholds and exponent formatting.
  defp python_float(value) do
    case String.split(:erlang.float_to_binary(value, [:short]), "e") do
      [fixed] ->
        fixed

      [mantissa, exponent] ->
        exponent = String.to_integer(exponent)

        if exponent < -4 or exponent >= 16 do
          sign = if exponent < 0, do: "-", else: "+"

          String.replace_suffix(mantissa, ".0", "") <>
            "e" <> sign <> String.pad_leading(Integer.to_string(abs(exponent)), 2, "0")
        else
          fixed_float(mantissa, exponent)
        end
    end
  end

  defp fixed_float(mantissa, exponent) do
    {sign, mantissa} =
      if String.starts_with?(mantissa, "-") do
        {"-", String.trim_leading(mantissa, "-")}
      else
        {"", mantissa}
      end

    digits = mantissa |> String.replace(".", "") |> String.trim_trailing("0")
    decimal_position = exponent + 1

    cond do
      decimal_position <= 0 ->
        sign <> "0." <> String.duplicate("0", -decimal_position) <> digits

      decimal_position >= byte_size(digits) ->
        sign <> digits <> String.duplicate("0", decimal_position - byte_size(digits)) <> ".0"

      true ->
        sign <>
          String.slice(digits, 0, decimal_position) <>
          "." <> String.slice(digits, decimal_position..-1//1)
    end
  end

  defp question_pairs(%Jason.OrderedObject{values: pairs}) do
    Enum.map(pairs, fn {id, question} -> {id, plain(question)} end)
  end

  defp question_pairs(pairs) when is_list(pairs), do: pairs
  defp question_pairs(map) when is_map(map), do: Enum.sort_by(map, &elem(&1, 0))

  defp question_pairs(_) do
    raise ArgumentError, "questions must be an object or ordered list"
  end

  defp options(%{"type" => "noul"} = question) do
    overrides = question["criteria"] || %{}

    unless is_map(overrides) do
      raise ArgumentError, "noul criteria must be an object"
    end

    criteria =
      Map.merge(
        %{
          "true" => "The proposition is true or the answer is yes.",
          "false" => "The proposition is false or the answer is no."
        },
        overrides
      )

    Enum.map(["true", "false"], &{&1, criteria[&1]})
  end

  defp options(%{"type" => "choice", "criteria" => criteria})
       when is_map(criteria) and map_size(criteria) > 0 do
    Enum.sort_by(criteria, &elem(&1, 0))
  end

  defp options(%{"type" => "score", "criteria" => criteria})
       when is_list(criteria) and criteria != [] do
    criteria
    |> Enum.with_index()
    |> Enum.map(fn {description, index} -> {Integer.to_string(index), description} end)
  end

  defp options(_) do
    raise ArgumentError, "choice/score questions require nonempty criteria"
  end

  defp shift_span({start, stop}, offset), do: {start + offset, stop + offset}
end
