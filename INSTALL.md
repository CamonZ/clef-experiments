# Installation guide for agents and humans

Run commands from the repository root. Download checkpoints separately; `models/`
is intentionally ignored by Git. Native serving uses Elixir and Metal, not Python.
Python is only needed for downloading with the method below or regenerating reference
fixtures. Do not substitute the original unconverted models or other quantizations:
these servers target the specific MLX conversions below.

## 1. Prerequisites

Use an Apple Silicon Mac with macOS 26.2 or later and Xcode command-line tools
(`xcode-select -p` should succeed). Install Elixir matching `mix.exs` (`~> 1.20`)
and a compatible Erlang/OTP. The existing installation used Elixir 1.20.4 / OTP 29.
Install Python 3 and create a downloader-only environment:

```sh
python3 -m venv .venv-download
.venv-download/bin/python -m pip install huggingface_hub
mix local.hex --force
mix local.rebar --force
mix deps.get
```

Allow roughly 13 GB of disk space for the two checkpoints, plus dependencies and
builds. Clef's measured warm resident MLX memory is about 9.31 GiB; KnowLine's is
about 2.69 GiB. Temporary inference allocations and concurrent requests need
additional unified memory. Four concurrent inferences is the default; use
`--max-concurrency 1` for initial smoke checks on a memory-constrained machine.

## 2. Download the required checkpoints

These revisions match the local checkpoints used to verify the port. Keep each
model in its own directory and rerun the same download to recover an interrupted
download. Do not mix files from different revisions in an existing directory.
Download only the model you need, or run both commands for the full test suite.

### Clef-Flash, MLX 8-bit

Source: [TrevorJS/clef-flash-mlx-8bit](https://huggingface.co/TrevorJS/clef-flash-mlx-8bit).

```sh
.venv-download/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TrevorJS/clef-flash-mlx-8bit",
    revision="2387dc48f97b94f1bc4ae66082fb8252e9463b6a",
    local_dir="models/clef-flash-8bit",
)
PY
```

Download the full snapshot, including both `model-*.safetensors` shards,
`joint_head.safetensors`, `joint_head_config.json`, `config.json`, and
`tokenizer.json`. The joint decision head is required in addition to the backbone.
Retain the checkpoint's license and metadata.

### KnowLine Gen5.5, MLX 5-bit

Source: [Liamariel/knowline-4b-gen5.5-mlx-q5](https://huggingface.co/Liamariel/knowline-4b-gen5.5-mlx-q5).

```sh
.venv-download/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Liamariel/knowline-4b-gen5.5-mlx-q5",
    revision="053081f764ffeddf4b1278d4183db95b72180520",
    local_dir="models/knowline-4b-gen5.5-mlx-q5",
)
PY
```

Download the full snapshot, including `model.safetensors`, `config.json`, and
`tokenizer.json`. KnowLine does not use Clef's separate joint head. Retain the
checkpoint's license and metadata. The known weight-file SHA-256 is:

```sh
shasum -a 256 models/knowline-4b-gen5.5-mlx-q5/model.safetensors
# 923bbeb8feff9578f2f094c6d2429d17e0095109f0fe8b98125f3b662231e654
```

The downloader retrieves files only; these commands do not import or execute
checkpoint Python code. Hugging Face documents revision-pinned downloads and
`local_dir` in its [download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download).

## 3. Build, test, and serve

```sh
mix compile --warnings-as-errors
mix test
mix format --check-formatted
```

The default tests need both downloaded tokenizers but do not load full model
weights. Model integration tests are opt-in; run only the model tests you installed:

```sh
CLEF_INTEGRATION=1 mix test test/integration_test.exs
KNOWLINE_INTEGRATION=1 mix test test/knowline_integration_test.exs
```

Start either server in a foreground terminal (separate terminals for both):

```sh
mix clef.serve --model models/clef-flash-8bit --max-concurrency 4
mix knowline.serve --model models/knowline-4b-gen5.5-mlx-q5 --max-concurrency 4
```

From another terminal, check the relevant server:

```sh
curl --fail http://127.0.0.1:46291/health
curl --fail http://127.0.0.1:46292/health
curl --fail http://127.0.0.1:46291/decide \
  -H 'Content-Type: application/json' \
  -d '{"state":"The database is unavailable.","questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"}}}'
```

Use port 46292 for the same KnowLine inference check. `--port` changes the listen
port; `--model` accepts an alternate checkpoint directory. Avoid starting a
foreground instance on a port already owned by a background service.

## 4. Optional macOS LaunchAgents

The checked-in `bin/*-server` launchers and `launchd/*.plist` currently contain
machine-specific project, toolchain, and log paths. Before installing on another
Mac, adapt these to its actual paths and verify its Elixir/OTP installation.
The service wrappers copy the plist as-is; they do not render portable templates.
Installation starts the service and enables login startup.

After adapting paths and completing foreground checks, stop the foreground server
and install whichever background services are wanted:

```sh
bin/clef-service install
bin/knowline-service install
bin/clef-service status
bin/knowline-service status
```

These wrappers use `launchctl` in the current user's GUI domain. Subsequent code
changes require a production rebuild before restarting:

```sh
MIX_ENV=prod mix compile --warnings-as-errors
bin/clef-service restart
bin/knowline-service restart
```

Use the matching wrapper's `logs` command to diagnose failures and `stop` to unload
weights and disable login startup. See README.md for API details, reference-fixture
regeneration, and existing model verification evidence.

## Publication boundary

Keep downloaded weights, downloader environments, caches, and generated artifacts
local. Preserve `NOTICE` and `licenses/` in the source repository. Before any
publication, inspect the staged file list and confirm no model files or credentials
are included. Installing checkpoints does not require Git LFS or uploading weights.
