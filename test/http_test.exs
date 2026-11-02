defmodule Clef.HTTPTest do
  use ExUnit.Case, async: true

  test "rejects malformed JSON and oversized bodies before invoking the model" do
    for body <- ["broken", "null", "[]", "{}", ~s({"questions":[]})] do
      conn =
        Plug.Test.conn(:post, "/decide", body)
        |> Clef.HTTP.call(:not_running)

      assert conn.status == 400
    end

    conn =
      Plug.Test.conn(:post, "/decide", String.duplicate("x", 1_000_001))
      |> Clef.HTTP.call(:not_running)

    assert conn.status == 413

    missing = Plug.Test.conn(:get, "/missing") |> Clef.HTTP.call(:not_running)
    assert missing.status == 404
  end
end
