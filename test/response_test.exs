defmodule Clef.ResponseTest do
  use ExUnit.Case, async: true

  test "wraps all question types with actual model identity and token usage" do
    record = %{
      "questions" => [
        {"implement", %{"type" => "noul"}},
        {"ticket", %{"type" => "choice"}},
        {"priority", %{"type" => "score", "criteria" => ["low", "high"]}}
      ]
    }

    response =
      Clef.Response.build(
        "clef-flash-8bit",
        record,
        %{
          "implement" => %{"true" => 0.9, "false" => 0.1},
          "ticket" => %{"unknown" => 0.2, "abc" => 0.8},
          "priority" => %{"0" => 0.25, "1" => 0.75}
        },
        123
      )

    assert response.model == "clef-flash-8bit"
    assert response.usage == %{input_tokens: 123, output_tokens: 0}
    assert response.answers["implement"] == %{type: "noul", noul: 0.9}

    assert response.answers["ticket"] == %{
             type: "choice",
             choice: "abc",
             confidence: 0.8,
             probabilities: %{"unknown" => 0.2, "abc" => 0.8}
           }

    assert response.answers["priority"] == %{
             type: "score",
             score: 0.75,
             confidence: 0.75,
             probabilities: %{"0" => 0.25, "1" => 0.75},
             legend: %{"0" => "low", "1" => "high"}
           }
  end
end
