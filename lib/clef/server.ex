defmodule Clef.Server do
  @moduledoc "Owns shared resident weights and bounds concurrent inference."

  use GenServer

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, Keyword.take(opts, [:name]))
  end

  def decide(server, record, opts \\ []) do
    GenServer.call(server, {:decide, record, opts}, :infinity)
  end

  def memory_info(server), do: GenServer.call(server, :memory_info)

  @impl true
  def init(opts) do
    Process.flag(:trap_exit, true)
    max_concurrency = Keyword.get(opts, :max_concurrency, 4)

    unless is_integer(max_concurrency) and max_concurrency > 0 do
      raise ArgumentError, "max_concurrency must be a positive integer"
    end

    module = Keyword.get(opts, :module, Clef)
    EMLX.set_cache_limit(Keyword.get(opts, :cache_limit, 256 * 1024 * 1024))
    model = module.load(Keyword.get(opts, :model_path, "models/clef-flash-8bit"))

    {:ok,
     %{
       module: module,
       model: model,
       max_concurrency: max_concurrency,
       running: %{},
       queue: :queue.new()
     }}
  end

  @impl true
  def handle_call(:memory_info, _from, state), do: {:reply, EMLX.memory_info(), state}

  def handle_call({:decide, record, opts}, from, state) do
    request = {from, record, opts}

    if map_size(state.running) < state.max_concurrency do
      {:noreply, start_request(state, request)}
    else
      {:noreply, %{state | queue: :queue.in(request, state.queue)}}
    end
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
    case Map.pop(state.running, ref) do
      {nil, _} ->
        {:noreply, state}

      {{from, _pid}, running} ->
        result =
          case reason do
            {:inference_result, result} -> result
            _ -> {:error, "inference worker failed"}
          end

        GenServer.reply(from, result)
        state = %{state | running: running}

        # Workers have exited, releasing their temporary tensor resources.
        # Only clear the process-wide allocator cache when inference is idle.
        if map_size(running) == 0, do: EMLX.clear_cache()

        case :queue.out(state.queue) do
          {{:value, request}, queue} ->
            {:noreply, start_request(%{state | queue: queue}, request)}

          {:empty, _} ->
            {:noreply, state}
        end
    end
  end

  def handle_info({:EXIT, _pid, _reason}, state), do: {:noreply, state}

  @impl true
  def terminate(_reason, state) do
    for {_ref, {_from, pid}} <- state.running, do: Process.exit(pid, :kill)
    :ok
  end

  defp start_request(state, {from, record, opts}) do
    module = state.module
    model = state.model

    {pid, ref} =
      :erlang.spawn_opt(
        fn ->
          result =
            try do
              {:ok, module.decide(model, record, opts)}
            rescue
              error in [ArgumentError, KeyError, FunctionClauseError] ->
                {:error, Exception.message(error)}
            end

          exit({:inference_result, result})
        end,
        [:link, :monitor]
      )

    %{state | running: Map.put(state.running, ref, {from, pid})}
  end
end
