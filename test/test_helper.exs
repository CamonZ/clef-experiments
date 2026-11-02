excluded = []

excluded =
  if System.get_env("CLEF_INTEGRATION") == "1", do: excluded, else: [:integration | excluded]

excluded =
  if System.get_env("KNOWLINE_INTEGRATION") == "1",
    do: excluded,
    else: [:knowline_integration | excluded]

ExUnit.start(exclude: excluded)
Clef.Native.load_plugin!()
