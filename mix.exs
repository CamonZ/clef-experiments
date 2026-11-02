defmodule Clef.MixProject do
  use Mix.Project

  def project do
    [
      app: :clef,
      version: "0.1.0",
      elixir: "~> 1.20",
      deps: deps(),
      compilers: [:elixir_make] ++ Mix.compilers()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:nx, "~> 1.0"},
      {:elixir_make, "~> 0.10", runtime: false},
      {:emlx,
       github: "elixir-nx/emlx", sparse: "emlx", ref: "10ddda6c1bf58d1f3b19f4e50d126dafe76db6d4"},
      {:jason, "~> 1.4"},
      {:bandit, "~> 1.0"},
      {:plug, "~> 1.0"},
      {:tokenizers, "~> 0.5"}
    ]
  end
end
