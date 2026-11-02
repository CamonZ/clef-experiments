defmodule Clef.InputTest do
  use ExUnit.Case, async: false

  setup_all do
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_file("models/clef-flash-8bit/tokenizer.json")
    {:ok, tokenizer: tokenizer}
  end

  test "matches the Python IDs and spans including ordered fields, Unicode, null and floats", %{
    tokenizer: tokenizer
  } do
    records = File.read!("fixtures/records.json") |> Jason.decode!(objects: :ordered_objects)
    reference = File.read!("fixtures/reference.json") |> Jason.decode!()

    for {record, expected} <- Enum.zip(records, reference) do
      record = record |> Jason.encode!() |> Clef.Input.decode!()
      actual = Clef.Input.encode(tokenizer, record)
      assert actual.ids == expected["ids"]
      assert Enum.map(actual.fields, & &1.id) == Enum.map(record["questions"], &elem(&1, 0))

      for {field, expected} <- Enum.zip(actual.fields, expected["fields"]) do
        assert Tuple.to_list(field.qspan) == expected["qspan"]
        assert Enum.map(field.ospans, &Tuple.to_list/1) == expected["ospans"]
        assert field.oids == expected["oids"]
        assert field.type == expected["type"]
      end
    end
  end

  test "truncates state without truncating the schema", %{tokenizer: tokenizer} do
    record = %{
      "state" => String.duplicate("hello ", 1000),
      "questions" => %{"q" => %{"type" => "noul"}}
    }

    actual = Clef.Input.encode(tokenizer, record, 160)
    assert length(actual.ids) == 160

    assert Enum.all?(actual.fields, fn field ->
             Enum.all?(field.ospans, fn {start, stop} -> start < stop and stop < 160 end)
           end)

    assert_raise ArgumentError, ~r/schema exceeds/, fn ->
      Clef.Input.encode(tokenizer, record, 10)
    end
  end

  test "rejects unsupported inputs before inference", %{tokenizer: tokenizer} do
    record = %{"state" => "ok", "questions" => %{"q" => %{"type" => "noul"}}}

    assert_raise ArgumentError, fn ->
      Clef.Input.encode(tokenizer, Map.put(record, "images", []))
    end

    assert_raise ArgumentError, fn ->
      Clef.Input.encode(tokenizer, Map.put(record, "questions", %{}))
    end

    assert_raise ArgumentError, fn ->
      Clef.Input.encode(tokenizer, %{
        record
        | "questions" => %{"q" => %{"type" => "choice", "criteria" => %{}}}
      })
    end

    assert_raise ArgumentError, fn ->
      Clef.Input.encode(tokenizer, %{record | "questions" => %{"q" => %{"type" => "unknown"}}})
    end

    for invalid <- ["[]", "null", "{}", ~s({"state":"ok","questions":[]})] do
      assert_raise ArgumentError, fn ->
        Clef.Input.decode!(invalid)
      end
    end
  end
end
