defmodule Mix.Tasks.Knowline.Serve do
  @shortdoc "Serve KnowLine Gen5.5 via native EMLX on localhost:46292"
  use Mix.Task

  def run(args) do
    {opts, [], []} =
      OptionParser.parse(args,
        strict: [port: :integer, model: :string, max_concurrency: :integer]
      )

    Mix.Task.run("app.start")
    port = Keyword.get(opts, :port, 46292)
    path = Keyword.get(opts, :model, "models/knowline-4b-gen5.5-mlx-q5")

    {:ok, server} =
      Clef.Server.start_link(
        module: Knowline,
        model_path: path,
        max_concurrency: Keyword.get(opts, :max_concurrency, 4)
      )

    {:ok, _http} =
      Bandit.start_link(
        plug: {Clef.HTTP, server: server, decoder: Knowline.Input},
        ip: {127, 0, 0, 1},
        port: port
      )

    IO.puts("KnowLine ready at http://127.0.0.1:#{port}; POST /v1/systemone or /decide")
    Process.sleep(:infinity)
  end
end
