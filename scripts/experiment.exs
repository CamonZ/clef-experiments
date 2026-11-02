model = Clef.load()

record = %{
  "state" => "The production database is unavailable and users cannot sign in.",
  "questions" => %{
    "urgent" => %{
      "type" => "noul",
      "instructions" => "Does this incident require immediate attention?"
    }
  }
}

model
|> Clef.decide(record)
|> Jason.encode!(pretty: true)
|> IO.puts()

IO.inspect(EMLX.memory_info(), label: "MLX memory (bytes)")
