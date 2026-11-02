# Clef experiments

Local decision-model servers for **Clef-Flash** and **KnowLine Gen5.5**, built
with Elixir, Nx, and EMLX on Apple Silicon. Inference runs natively on Metal;
Python is only needed for downloading checkpoints or generating reference fixtures.

See [INSTALL.md](INSTALL.md) for prerequisites, pinned model downloads, and setup.
Weights are downloaded separately and are excluded from this repository.

## Run

After installation, start either server:

```sh
mix clef.serve
# http://127.0.0.1:46291

mix knowline.serve
# http://127.0.0.1:46292
```

Each server loads its model once and shares the weights across requests. The
default is four concurrent inferences; additional requests wait in a FIFO queue.
Use `--max-concurrency`, `--port`, and `--model` to override settings:

```sh
mix clef.serve --max-concurrency 2 --port 46293 --model models/clef-flash-8bit
```

Higher concurrency needs more temporary memory. The waiting queue is currently
unlimited. The MLX allocator cache is capped at 256 MiB, and unused cached buffers
are cleared when inference is idle; this is not a limit on total memory use.

## API

Both servers accept `POST /decide` and `POST /v1/systemone`.
`GET /health` reports readiness.

```sh
curl http://127.0.0.1:46291/decide \
  -H 'Content-Type: application/json' \
  -d '{"state":"The database is down and nobody can sign in.","questions":{"urgent":{"type":"noul","instructions":"Does this require immediate attention?"}}}'
```

Use port 46292 for KnowLine. Requests contain `state` and a `questions` object:

- `noul`: true/false probability, with optional descriptions in `criteria`.
- `choice`: options mapping IDs to descriptions.
- `score`: an ordered list of level descriptions.

Responses contain `model`, `answers`, and `usage`. The model identifies the
loaded checkpoint; sending a model alias in a request does not switch weights.
Answers include the true probability, selected choice, or expected score.
Clef confidence is the largest option probability; KnowLine confidence is
normalized inverse entropy. Invalid records return HTTP 400; request bodies
over 1 MB return HTTP 413.

These are text/JSON inference ports for the specific MLX conversions in the
installation guide. Clef truncates state to fit its default 16,384-token limit.
KnowLine rejects prompts above 16,384 tokens and supports up to 64 questions
with up to 255 options each; choice and score require at least two options.
JSON question order is preserved. Plain Elixir maps sort keys deterministically.

## Elixir

```elixir
{:ok, server} = Clef.Server.start_link(max_concurrency: 4)

record = %{
  "state" => "The production database is unavailable.",
  "questions" => %{"urgent" => %{"type" => "noul", "instructions" => "Is this urgent?"}}
}

{:ok, probabilities} = Clef.Server.decide(server, record)
```

For KnowLine, pass `module: Knowline` and
`model_path: "models/knowline-4b-gen5.5-mlx-q5"` to `start_link/1`.
`Clef.Server.memory_info/1` reports active, peak, and cached MLX memory.

## Background services

Both servers include macOS LaunchAgent wrappers. Adapt the machine-specific
paths first, following [INSTALL.md](INSTALL.md#4-optional-macos-launchagents).

```sh
bin/clef-service install
bin/knowline-service install
```

Each wrapper supports `status`, `start`, `stop`, `restart`, and `logs`.
Services start at login and restart after exit. `stop` unloads the model and
disables login startup until `start` is called. Rebuild with
`MIX_ENV=prod mix compile` before restarting after code changes.

## Tests

```sh
mix test
CLEF_INTEGRATION=1 mix test test/integration_test.exs
KNOWLINE_INTEGRATION=1 mix test test/knowline_integration_test.exs
mix format --check-formatted
```

Default tests require both local tokenizers; integration tests load model weights.

See [NOTICE](NOTICE) and [licenses/](licenses/) for third-party attribution.
