defmodule Clef.IntegrationTest do
  use ExUnit.Case, async: false

  @moduletag :integration
  @moduletag timeout: 180_000

  test "resident model matches Python probabilities and starts each query with fresh state" do
    server = start_supervised!({Clef.Server, max_concurrency: 2})
    records = File.read!("fixtures/records.json") |> Jason.decode!(objects: :ordered_objects)
    expected = File.read!("fixtures/reference.json") |> Jason.decode!()

    results =
      Enum.zip(records, expected)
      |> Enum.map(fn {record, expected} ->
        conn =
          Plug.Test.conn(:post, "/decide", Jason.encode!(record))
          |> Clef.HTTP.call(server)

        assert conn.status == 200
        envelope = Jason.decode!(conn.resp_body)
        assert envelope["model"] == "clef-flash-8bit"
        assert envelope["usage"]["input_tokens"] > 0
        assert envelope["usage"]["output_tokens"] == 0
        decoded = record |> Jason.encode!() |> Clef.Input.decode!()
        assert {:ok, result} = Clef.Server.decide(server, decoded)

        expected_envelope =
          Clef.Response.build(
            "clef-flash-8bit",
            decoded,
            result,
            envelope["usage"]["input_tokens"]
          )
          |> Jason.encode!()
          |> Jason.decode!()

        assert envelope == expected_envelope

        for {id, options} <- result do
          assert_in_delta Enum.sum(Map.values(options)), 1.0, 1.0e-6

          {top_option, _} = Enum.max_by(options, &elem(&1, 1))

          {expected_option, _} =
            Enum.max_by(expected["probabilities"][id], &elem(&1, 1))

          assert top_option == expected_option

          for {option_id, probability} <- options do
            assert_in_delta probability, expected["probabilities"][id][option_id], 0.005
          end
        end

        result
      end)

    record = hd(records) |> Jason.encode!() |> Clef.Input.decode!()
    assert {:ok, hd(results)} == Clef.Server.decide(server, record)

    concurrent =
      1..4
      |> Task.async_stream(fn _ -> Clef.Server.decide(server, record) end,
        max_concurrency: 4,
        timeout: 30_000
      )
      |> Enum.to_list()

    assert concurrent == List.duplicate({:ok, {:ok, hd(results)}}, 4)

    invalid =
      Plug.Test.conn(:post, "/decide", ~s({"state":"ok","questions":{}}))
      |> Clef.HTTP.call(server)

    assert invalid.status == 400

    invalid =
      Plug.Test.conn(
        :post,
        "/decide",
        ~s({"state":"ok","questions":{"q":{"type":"noul","criteria":[]}}})
      )
      |> Clef.HTTP.call(server)

    assert invalid.status == 400
    assert Process.alive?(server)

    health = Plug.Test.conn(:get, "/health") |> Clef.HTTP.call(server)
    assert health.status == 200
    assert {:ok, hd(results)} == Clef.Server.decide(server, record)
  end
end
