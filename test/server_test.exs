defmodule Clef.ServerTest do
  use ExUnit.Case, async: false

  defmodule Model do
    def load(owner), do: owner

    def decide(owner, id, _opts) do
      send(owner, {:started, id, self()})

      receive do
        :finish -> id
        :invalid -> raise ArgumentError, "invalid request"
      end
    end
  end

  test "shares one model, limits active requests and drains the queue after errors" do
    server =
      start_supervised!({Clef.Server, module: Model, model_path: self(), max_concurrency: 2})

    first = Task.async(fn -> Clef.Server.decide(server, 1) end)
    assert_receive {:started, 1, worker1}
    second = Task.async(fn -> Clef.Server.decide(server, 2) end)
    assert_receive {:started, 2, worker2}
    third = Task.async(fn -> Clef.Server.decide(server, 3) end)
    refute_receive {:started, 3, _}
    send(worker1, :invalid)
    assert Task.await(first) == {:error, "invalid request"}
    assert_receive {:started, 3, worker3}
    send(worker2, :finish)
    send(worker3, :finish)
    assert Task.await(second) == {:ok, 2}
    assert Task.await(third) == {:ok, 3}
  end

  test "defaults to four active requests" do
    server = start_supervised!({Clef.Server, module: Model, model_path: self()})

    active =
      for id <- 1..4 do
        task = Task.async(fn -> Clef.Server.decide(server, id) end)
        assert_receive {:started, ^id, worker}
        {id, task, worker}
      end

    waiting = Task.async(fn -> Clef.Server.decide(server, 5) end)
    refute_receive {:started, 5, _}
    [{_, _, first_worker} | _] = active
    send(first_worker, :finish)
    assert_receive {:started, 5, fifth_worker}
    send(fifth_worker, :finish)

    for {id, task, worker} <- active do
      send(worker, :finish)
      assert Task.await(task) == {:ok, id}
    end

    assert Task.await(waiting) == {:ok, 5}
  end
end
