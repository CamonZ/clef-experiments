defmodule Mix.Tasks.Clef.Serve do
  @shortdoc "Load Clef once and serve queries on localhost:46291"

  use Mix.Task

  @impl true
  def run(args) do
    {opts, [], []} =
      OptionParser.parse(args,
        strict: [port: :integer, model: :string, max_concurrency: :integer]
      )

    Mix.Task.run("app.start")

    port = Keyword.get(opts, :port, 46291)
    model_path = Keyword.get(opts, :model, "models/clef-flash-8bit")

    IO.puts("Loading #{model_path} on Metal…")

    {:ok, server} =
      Clef.Server.start_link(
        model_path: model_path,
        max_concurrency: Keyword.get(opts, :max_concurrency, 4)
      )

    {:ok, _http} =
      Bandit.start_link(plug: {Clef.HTTP, server: server}, ip: {127, 0, 0, 1}, port: port)

    IO.puts(
      "Clef ready at http://127.0.0.1:#{port}; weights stay loaded until this process stops."
    )

    Process.sleep(:infinity)
  end
end
