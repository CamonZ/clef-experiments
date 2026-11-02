defmodule Clef.Response do
  @moduledoc "TypeSafe-compatible HTTP response for the resident Clef checkpoint."

  def build(model, record, probabilities, input_tokens) do
    answers =
      Map.new(record["questions"], fn {id, question} ->
        {id, answer(question, Map.fetch!(probabilities, id))}
      end)

    %{model: model, answers: answers, usage: %{input_tokens: input_tokens, output_tokens: 0}}
  end

  defp answer(%{"type" => "noul"}, probabilities) do
    %{type: "noul", noul: Map.fetch!(probabilities, "true")}
  end

  defp answer(%{"type" => "choice"}, probabilities) do
    {choice, confidence} = probabilities |> Enum.sort() |> Enum.max_by(&elem(&1, 1))
    %{type: "choice", choice: choice, probabilities: probabilities, confidence: confidence}
  end

  defp answer(%{"type" => "score", "criteria" => criteria}, probabilities) do
    score =
      Enum.reduce(probabilities, 0.0, fn {level, probability}, total ->
        total + String.to_integer(level) * probability
      end)

    legend =
      criteria
      |> Enum.with_index()
      |> Map.new(fn {description, index} ->
        {Integer.to_string(index), Clef.Input.render(description)}
      end)

    %{
      type: "score",
      score: score,
      probabilities: probabilities,
      confidence: probabilities |> Map.values() |> Enum.max(),
      legend: legend
    }
  end
end
