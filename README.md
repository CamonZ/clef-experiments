# Clef experiments

For a fresh checkout, follow [INSTALL.md](INSTALL.md): agent-ready prerequisites,
pinned downloads for both checkpoints, tests, serving, and LaunchAgent setup.

Native Elixir/Nx inference for the downloaded **TrevorJS Clef-Flash MLX 8-bit**
checkpoint. EMLX runs Nx operations on Apple's Metal GPU. A small C++/Metal
plugin implements Qwen3.5's gated recurrence; the decision head runs in float32.
Inference does not start Python. The existing Python experiment is kept as an
independent verification reference.

## Run the resident model

From this folder:

```sh
mix clef.serve
```

It loads the model once and serves on **127.0.0.1:46291**. Leave that terminal
running; subsequent requests reuse the weights. Ctrl-C stops the process.
Each request starts with fresh attention/recurrent state. By default, up to four requests
run concurrently. Set `--max-concurrency 2` on either `mix clef.serve` or
`mix knowline.serve` to allow two simultaneous inferences sharing resident weights.
The value must be a positive integer; excess requests wait in a FIFO queue.
Direct callers can use `Clef.Server.start_link(max_concurrency: 2)`. Higher
concurrency increases temporary memory use and may not improve GPU throughput.
The waiting queue is not currently size-limited.

```sh
curl http://127.0.0.1:46291/decide \
  -H 'Content-Type: application/json' \
  -d '{"state":"The database is down and nobody can sign in.","questions":{"urgent":{"type":"noul","instructions":"Does this require immediate attention?"}}}'
```

The HTTP response is a TypeSafe-compatible envelope with `model`, `answers`,
and `usage`. `model` identifies the loaded checkpoint directory (normally
`clef-flash-8bit`); a request's optional `model` alias does not select or rename
weights. Noul answers contain `type` and `noul` (the probability of true).
Choice answers contain `type`, `choice`, `probabilities`, and `confidence`.
Score answers contain `type`, `score` (expected level), `legend`,
`probabilities`, and `confidence`. Confidence is the largest option probability.
Usage reports the actual encoded input length after truncation and zero output
tokens, because the decision head does not generate text. Direct Elixir calls
still return raw option probabilities.

`GET /health`
returns readiness. Invalid records return HTTP 400; bodies over 1 MB return 413.
Use `mix clef.serve --port 46292 --model models/clef-flash-8bit` to change settings.

## macOS background service

This Mac has a user LaunchAgent named `com.camonz.clef`, serving the same API
on **127.0.0.1:46291**. It keeps the 8-bit model resident, starts at login and
restarts after process exits. It runs without a terminal or Python environment.
The launcher uses the installed Elixir/OTP paths explicitly and starts an
already-compiled production build, so launchd does not compile on each restart.

```sh
bin/clef-service status
bin/clef-service stop
bin/clef-service start
bin/clef-service restart
bin/clef-service logs
```

`stop` also disables startup at login; `start` re-enables it. Stopping unloads
the weights and releases their memory. Avoid starting the foreground server
while the service owns port 46291. This port is below this Mac's automatic
client-port range (49152–65535) and avoids common development-server defaults.

The installed configuration is
`~/Library/LaunchAgents/com.camonz.clef.plist`, with logs in
`~/Library/Logs/Clef/`. The source configuration is in `launchd/`. Its project
path and `bin/clef-server`'s toolchain paths are specific to this machine.
To install again, run `bin/clef-service install`. After changing Elixir code or
native kernels, run `MIX_ENV=prod mix compile` and `bin/clef-service restart`.

## Call from Elixir

```sh
iex -S mix
```

```elixir
{:ok, server} = Clef.Server.start_link()
record = %{
  "state" => "The production database is unavailable.",
  "questions" => %{"urgent" => %{"type" => "noul", "instructions" => "Is this urgent?"}}
}
{:ok, probabilities} = Clef.Server.decide(server, record)
```

`Clef.load/1` and `Clef.decide/3` also work directly. A one-shot smoke run is
`mix run scripts/experiment.exs`.

## Inputs and scope

- `noul`: true/false, with optional descriptions in `criteria`.
- `choice`: a nonempty object mapping option IDs to descriptions.
- `score`: a nonempty list of descriptions, assigned IDs `"0"`, `"1"`, etc.
- `state` and `instructions`: text or JSON. Choice options sort by ID.
- JSON requests preserve question order. Elixir maps sort question IDs; use an
  ordered list of `{id, question}` pairs, or `Clef.Input.decode!/1`, to preserve
  an explicit order. Field order affects the joint input.
- The default input limit is 16,384 tokens. The state is truncated to fit;
  an oversized schema raises an error. Direct Elixir callers can pass
  `max_length: n` to `decide`. Long inputs have not been benchmarked.

This is an experimental **text/JSON-only inference port** for this specific
converted 8-bit checkpoint. It does not include the vision encoder, training,
the full Clef model, other quantizations, streaming text generation, or cached
multi-turn decoding. The custom recurrence is eager EMLX; the entire model has
not been wrapped in an Nx.Defn compiled graph.

## Setup on this machine

Elixir 1.20.4 / OTP 29, Xcode, model weights and native dependencies are already
installed, and the project is compiled. No virtual-environment activation or
LM Studio is needed for native inference.

For another compatible Apple Silicon Mac, install Elixir/OTP and Apple's Xcode
command-line tools, put the converted checkpoint in `models/clef-flash-8bit`,
and run `mix local.hex --force`, `mix local.rebar --force`, then `mix deps.get`.
`mix compile` builds the plugin and EMLX fetches its pinned MLX runtime. The
current dependency requires macOS 26.2+; this setup was tested on macOS 27.2.
EMLX is pinned to commit `10ddda6c1bf58d1f3b19f4e50d126dafe76db6d4` in `mix.exs`;
`mix.lock` pins the other dependencies.

## Verification

Observed on this Apple M5 Max (48 GB) using three reference records (154–358
tokens, seven decisions): all top choices agreed with Python MLX; maximum
absolute probability difference was **0.00172**. With identical backbone hidden
states, the head's maximum probability difference was **2.4e-7**. Embeddings,
token IDs and field spans matched exactly. The native Metal recurrence also
matched the independent MLX operations fixture within **1e-6**.

Checkpoint loading took about **1.6–2.1 seconds**, and warm sample queries took
about **0.2 seconds**. These are small local experiments, not a broad accuracy
or performance benchmark. Backbone outputs are not bit-identical to Python;
long-context behavior and wider task accuracy still need evaluation.

```sh
mix test
CLEF_INTEGRATION=1 mix test
mix format --check-formatted
```

The default suite uses the local tokenizer and small golden fixtures without
loading the model. The integration test loads it and checks all three question
types, repeated queries through the resident server and recovery after invalid
input. Golden fixtures are generated by the independent Python MLX implementation:

```sh
uv run python scripts/reference.py
mix run scripts/verify.exs
```

The verification script compares token IDs/spans, embeddings, intermediate
backbone layers and head probabilities on identical hidden states. It requires
the reference artifacts generated by the preceding command. Golden fixture
generation uses Python only for verification; these commands do not affect the
native serving path. See `NOTICE` and `licenses/` for source attribution.

## KnowLine Gen5.5 alongside Clef

The native Elixir server also supports the text-only community conversion
`Liamariel/knowline-4b-gen5.5-mlx-q5` (Apache-2.0, approximately 2.89 GB of
5-bit weights). Download it without changing the Clef background service:

```sh
.venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("Liamariel/knowline-4b-gen5.5-mlx-q5", revision="053081f764ffeddf4b1278d4183db95b72180520", local_dir="models/knowline-4b-gen5.5-mlx-q5")'
mix knowline.serve
```

KnowLine listens on **127.0.0.1:46292** by default. Both `POST /v1/systemone`
and `POST /decide` accept the state/questions request shape shown above;
`GET /health` reports readiness. `--model` and `--port` override the defaults.
No Python process participates in native inference.

The same Qwen3.5 backbone and Metal recurrence are reused. KnowLine bypasses
Clef's joint head: it evaluates each question separately and scores only the
allowed option-label embeddings against the last hidden state. Five-bit packed
weights go through EMLX's raw quantized operations, avoiding unsupported Nx
five-bit integer types. Calibration is already folded into the checkpoint;
no additional temperature is applied.

KnowLine preserves JSON question/option order, formats structured state as the
publisher's indented text, and repeats short requests using the Gen5.5
read-twice layout (whole-request read-once token count at most 3,000).
`Knowline.Input.decode!/1` preserves nested JSON order for direct Elixir calls;
plain Elixir maps sort keys deterministically. Supported chat states use
system/user/assistant roles and text content. Media and tool-call conversations
are rejected. The maximum is 64 questions and 255 options per question; choice
and score require at least two options. Prompts over 16,384 tokens are rejected
rather than truncated. Direct callers can set `max_length`, `read_twice` and
`twice_max_tokens` options through `Clef.Server.decide/3` or `Knowline.decide/3`.

KnowLine confidence is normalized inverse entropy, matching its reference
server; Clef confidence remains maximum option probability. KnowLine usage
counts tokens across question prompts and reports one output label per question,
matching the publisher's usage convention (the native implementation does not
generate those labels). The response model identifies the loaded directory.

New resident servers cap the process-wide MLX allocator cache at 256 MiB,
release temporary tensor resources when each request worker exits and clear unused
cached buffers when no requests are active. `Clef.Server.memory_info(server)` exposes active, peak and cached
bytes. Direct model calls should configure/clear their own cache. These source
changes do not alter the already-running Clef process until it is restarted.

### KnowLine verification

The reference script uses the publisher's MIT-licensed renderer at revision
`779944d2cc8ffa19cef386aee83c83bcb967ade2` and the independent Python mlx-lm
backbone. Generated fixtures record renderer source hashes. Python is needed
only to regenerate fixtures; the runtime and ExUnit checks use Elixir.
Use MLX **0.32.0** for reference generation, matching the installed EMLX runtime.
The original Python environment has 0.32.3; even that patch difference can move
BF16 decision probabilities. Run the reference in an isolated pinned environment:

```sh
uv run --no-project --with mlx==0.32.0 --with mlx-lm==0.31.3 --with requests==2.34.2 --with transformers==5.18.0 python scripts/knowline_reference.py --prompts-only
uv run --no-project --with mlx==0.32.0 --with mlx-lm==0.31.3 --with requests==2.34.2 --with transformers==5.18.0 python scripts/knowline_reference.py
```

```sh
mix test
KNOWLINE_INTEGRATION=1 mix test test/knowline_integration_test.exs
CLEF_INTEGRATION=1 mix test test/integration_test.exs
mix format --check-formatted
```

The default tests compare exact prompt bytes, token IDs, option-label IDs and
read-twice boundary behavior, plus a small independent five-bit matmul check.
The optional integration check loads KnowLine, compares seven decisions across
four records to Python MLX, exercises both request validation and the HTTP
adapter, and checks repeated-request memory. This validates the port, not the
Decision Index accuracy of this community quantization.

Observed on this machine: all seven decisions agreed with Python MLX 0.32.0;
maximum absolute probability difference was below `1e-6`. The final hidden
states and selected-label logits also matched exactly in the diagnostic run.
Repeated three-question requests took approximately 216–226 ms, with 2.89 GB
active MLX memory and zero allocator-cache bytes after each request. A separate
TCP server smoke check returned the expected envelope in 252 ms and reported
3.1 GiB process footprint (3.9 GiB peak). These are small fixture measurements,
not a Decision Index evaluation or a general latency guarantee.

The downloaded `model.safetensors` SHA-256 is
`923bbeb8feff9578f2f094c6d2429d17e0095109f0fe8b98125f3b662231e654`,
matching the Hugging Face file digest.

### KnowLine macOS background service

Install the compiled native server as the user LaunchAgent `com.camonz.knowline`:

```sh
bin/knowline-service install
bin/knowline-service status
bin/knowline-service logs
bin/knowline-service stop
bin/knowline-service start
bin/knowline-service restart
```

It serves `http://127.0.0.1:46292`, starts at login, and restarts after exiting.
`stop` unloads the model and disables login startup until `start` is called.
The launcher runs the already-compiled production build without compiling at
login. After code or kernel changes, run `MIX_ENV=prod mix compile` before
restarting. Clef uses its separate service and port 46291.

The installed plist is `~/Library/LaunchAgents/com.camonz.knowline.plist`;
stdout and stderr are in `~/Library/Logs/KnowLine/`. Source files are
`launchd/com.camonz.knowline.plist`, `bin/knowline-server`,
`bin/knowline-service`, and `scripts/knowline_daemon.exs`. As with the Clef
service, the launcher/toolchain and plist paths are specific to this Mac.
Do not start a foreground KnowLine server while the background service owns
port 46292.
