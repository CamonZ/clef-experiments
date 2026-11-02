import Config

config :nx, :default_backend, {EMLX.Backend, device: :gpu}
config :nx, :default_defn_options, compiler: EMLX
