defmodule Clef.HTTP do
  @moduledoc "Local JSON API. POST /decide with a Clef record; GET /health for readiness."

  import Plug.Conn

  def init(opts), do: {Keyword.fetch!(opts, :server), Keyword.get(opts, :decoder, Clef.Input)}

  def call(%{method: "GET", request_path: "/health"} = conn, _server) do
    respond(conn, 200, %{status: "ready"})
  end

  def call(conn, server) when not is_tuple(server), do: call(conn, {server, Clef.Input})

  def call(%{method: "POST", request_path: path} = conn, {server, decoder})
      when path in ["/decide", "/v1/systemone"] do
    case read_body(conn, length: 1_000_000, read_length: 1_000_000) do
      {:ok, body, conn} ->
        try do
          record = decoder.decode!(body)

          case Clef.Server.decide(server, record, envelope: true) do
            {:ok, result} ->
              respond(conn, 200, result)

            {:error, error} ->
              respond(conn, 400, %{error: error})
          end
        rescue
          error in [ArgumentError, Jason.DecodeError, KeyError, FunctionClauseError] ->
            respond(conn, 400, %{error: Exception.message(error)})
        end

      {:more, _body, conn} ->
        respond(conn, 413, %{error: "request body exceeds 1 MB"})

      {:error, _reason} ->
        respond(conn, 400, %{error: "could not read request body"})
    end
  end

  def call(conn, _server) do
    respond(conn, 404, %{error: "not found"})
  end

  defp respond(conn, status, value) do
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(status, Jason.encode!(value))
  end
end
