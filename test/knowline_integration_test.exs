defmodule Knowline.IntegrationTest do
  use ExUnit.Case, async: false
  @moduletag :knowline_integration
  @moduletag timeout: 300_000

  test "native probabilities agree with Python MLX; HTTP recovers and request memory stays bounded" do
    server =
      start_supervised!(
        {Clef.Server, module: Knowline, model_path: "models/knowline-4b-gen5.5-mlx-q5"}
      )

    reference =
      File.read!("fixtures/knowline_reference.json") |> Jason.decode!(objects: :ordered_objects)

    assert reference["runtime"]["mlx"] == "0.32.0"

    for entry <- reference["records"] do
      record = entry["record"]
      assert {:ok, probabilities} = Clef.Server.decide(server, record)

      for row <- entry["rows"] do
        actual = probabilities[row["id"]]
        expected = Knowline.Input.pairs(row["probabilities"]) |> Map.new()
        assert Enum.max_by(row["keys"], &actual[&1]) == Enum.max_by(row["keys"], &expected[&1])
        assert_in_delta Enum.sum(Map.values(actual)), 1.0, 1.0e-6
        for {key, probability} <- expected, do: assert_in_delta(actual[key], probability, 1.0e-6)
      end
    end

    record = hd(reference["records"])["record"]
    options = Clef.HTTP.init(server: server, decoder: Knowline.Input)
    body = Jason.encode!(record)
    before_memory = Clef.Server.memory_info(server)

    snapshots =
      for _ <- 1..5 do
        {elapsed, conn} =
          :timer.tc(fn ->
            Plug.Test.conn(:post, "/v1/systemone", body) |> Clef.HTTP.call(options)
          end)

        assert conn.status == 200
        envelope = Jason.decode!(conn.resp_body)
        assert envelope["model"] == "knowline-4b-gen5.5-mlx-q5"
        assert envelope["usage"]["output_tokens"] == 3
        assert envelope["answers"]["urgent"]["noul"] > 0.5

        for row <- hd(reference["records"])["rows"] do
          actual_answer = envelope["answers"][row["id"]]
          expected_answer = row["answer"] |> Jason.encode!() |> Jason.decode!()

          for {key, value} <- expected_answer do
            cond do
              is_float(value) ->
                assert_in_delta actual_answer[key], value, 1.0e-6

              key == "probabilities" ->
                for {option, p} <- value,
                    do: assert_in_delta(actual_answer[key][option], p, 1.0e-6)

              true ->
                assert actual_answer[key] == value
            end
          end
        end

        info = Clef.Server.memory_info(server)
        IO.inspect(%{milliseconds: elapsed / 1000, memory: info}, label: "KnowLine warm request")
        assert info.cache_memory <= 256 * 1024 * 1024
        assert info.active_memory < before_memory.active_memory + 128 * 1024 * 1024
        info
      end

    assert List.last(snapshots).active_memory < before_memory.active_memory + 128 * 1024 * 1024

    invalid =
      Plug.Test.conn(
        :post,
        "/decide",
        ~s({"state":"x","questions":{"q":{"type":"choice","criteria":{}}}})
      )
      |> Clef.HTTP.call(options)

    assert invalid.status == 400
    malformed = Plug.Test.conn(:post, "/decide", "null") |> Clef.HTTP.call(options)
    assert malformed.status == 400
    assert Process.alive?(server)
    assert (Plug.Test.conn(:post, "/decide", body) |> Clef.HTTP.call(options)).status == 200
  end
end
