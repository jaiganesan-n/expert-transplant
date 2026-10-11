"""`transplant ask`: the one-sentence question-and-answer test and the comparison of runs."""
import json

import pytest

from transplant.eval.ask import (EXAMPLES, SCORING, Question, _copied, _correct, answer_forms, answer_logprob, ask,
                                 best_answer_logprob, compare, default_questions, example_indices, fact_recovery,
                                 format_markdown, greedy_batch, load_questions, qa_prompt, reference_sentence,
                                 split_reference)
from transplant.eval.samples import greedy
from transplant.models.load import build_hybrid


def test_batched_greedy_equals_one_prompt_at_a_time(tiny_cfg, cpu, fp32_policy):
    hybrid = build_hybrid(tiny_cfg, cpu, fp32_policy)
    tok, model = hybrid.tokenizer, hybrid.model
    prompts = ["the cat sat", "a long time ago in a far away place there", "one two"]
    batched = greedy_batch(model, tok, prompts, 6, cpu, stop_at_newline=False)
    for p, b in zip(prompts, batched):
        alone = greedy(model, tok, p, 6, cpu, length=64)
        assert alone.endswith(b.strip()) or b.strip() == ""


def test_answer_logprob_is_a_mean_log_probability(tiny_cfg, cpu, fp32_policy):
    hybrid = build_hybrid(tiny_cfg, cpu, fp32_policy)
    lps = answer_logprob(hybrid.model, hybrid.tokenizer, ["the cat", "a dog"], ["sat", ""], cpu)
    assert lps[1] is None and lps[0] < 0


def test_answer_forms_and_best_logprob(tiny_cfg, cpu, fp32_policy):
    assert answer_forms(["seven", "7"]) == ["seven", "Seven", "7"]
    assert answer_forms(["H2O", "carbon dioxide"]) == ["H2O", "h2o", "carbon dioxide", "Carbon dioxide"]
    hybrid = build_hybrid(tiny_cfg, cpu, fp32_policy)
    tok, model = hybrid.tokenizer, hybrid.model
    best = best_answer_logprob(model, tok, ["the cat", "a dog"], [["sat", "Sat"], []], cpu)
    each = answer_logprob(model, tok, ["the cat", "the cat"], ["sat", "Sat"], cpu)
    assert best[1] is None and abs(best[0] - max(each)) < 1e-5


def test_references_and_the_built_in_questions():
    assert split_reference("The capital of France is [Paris].") == ("The capital of France is", "Paris", ".")
    assert split_reference("Water freezes at [0] degrees Celsius.") == ("Water freezes at", "0", " degrees Celsius.")
    assert reference_sentence("There are [seven] days in a week.") == "There are seven days in a week."
    qs = default_questions()
    assert len(qs) == 30 and len({q.text for q in qs}) == 30
    example_questions = {q for q, _ in EXAMPLES}
    for q in qs:
        stem, fact, _ = split_reference(q.reference)
        assert stem and fact and q.answers[0] == fact and q.topic
        assert q.text not in example_questions
        # a key answer never appears in the question, so copying the question cannot score
        assert not _correct(q.text, q.answers), q.text
        assert _correct(reference_sentence(q.reference), q.answers), q.reference


def test_examples_rotate_from_question_to_question():
    assert example_indices(0) == [0, 1, 2] and example_indices(5) == [5, 0, 1]
    p0, p1 = qa_prompt("What is the capital of France?", 0), qa_prompt("What is the capital of France?", 1)
    assert p0 != p1 and p0.endswith("Q: What is the capital of France?\nA:")
    assert p0.startswith("Answer each question with one complete sentence.")
    assert p0.count("\nA: ") == 3 and EXAMPLES[2][1] in p0 and EXAMPLES[2][1] in p1 and EXAMPLES[0][1] not in p1


def test_correct_matches_whole_words_and_copies_are_flagged():
    assert _correct("Water freezes at 0 degrees Celsius.", ["0", "zero"])
    assert not _correct("Water freezes at 100 degrees Celsius.", ["0", "zero"])
    assert _correct("The capital of France is Paris.", ["Paris"]) and not _correct("Parisian food.", ["Paris"])
    assert _correct("It was written by William Shakespeare.", ["William Shakespeare", "Shakespeare"])
    assert _correct("anything", []) is None
    # question 0 is shown examples 0, 1, 2; example 3 is not shown with it
    assert _copied("Grass is green.", "What is the capital of France?", 0)
    assert _copied("what is the capital of France", "What is the capital of France?", 0)
    assert not _copied(EXAMPLES[3][1], "What is the capital of France?", 0)
    assert not _copied("The capital of France is Paris.", "What is the capital of France?", 0)


def test_question_file_formats(tmp_path):
    f = tmp_path / "q.txt"
    f.write_text("# comment\n"
                 "What is the capital of Spain? | The capital of Spain is [Madrid]. | madrid\n"
                 "What is 3 + 4? | 7 | seven\n"
                 "Name a colour\n"
                 '{"question": "2+3?", "reference": "2 + 3 equals [5].", "answers": ["five"], "topic": "math"}\n')
    qs = load_questions(str(f))
    assert [q.answers for q in qs] == [["Madrid", "madrid"], ["7", "seven"], [], ["5", "five"]]
    assert [q.reference for q in qs] == ["The capital of Spain is [Madrid].", "", "", "2 + 3 equals [5]."]
    assert qs[3].topic == "math"
    assert len(load_questions(None)) == 30


def test_ask_reports_every_mode(tiny_cfg, cpu, fp32_policy):
    hybrid = build_hybrid(tiny_cfg, cpu, fp32_policy)
    qs = [Question("what is the cat", ["sat"], "the cat [sat] down.", "animals"), Question("say one", []),
          Question("count", ["two"])]
    res = ask(hybrid, qs, cpu, modes=("gemma", "hybrid", "ablated"), max_new_tokens=4, log=lambda *a: None)
    assert res["scoring"] == SCORING and res["format"] == "sentence"
    for m in ("gemma", "hybrid", "ablated"):
        r = res["modes"][m]
        assert len(r["answers"]) == 3 and r["correct"][1] is None and r["accuracy"] in (0.0, 0.5, 1.0)
        assert r["fact_logprob"][0] < 0 and r["fact_logprob"][1] is None and r["fact_logprob"][2] < 0
        assert r["sentence_logprob"][0] < 0 and r["sentence_logprob"][1] is None
        assert isinstance(r["copies"], int)
    md = format_markdown(res, "(tiny)")
    assert "| Gemma |" in md and "| Hybrid |" in md and "| No FFN |" in md and "## Replies" in md
    assert "**1. what is the cat** (animals; expected: the cat sat down.)" in md
    json.dumps(res)


def test_fact_logprob_is_scored_after_the_reference_opening(tiny_cfg, cpu, fp32_policy):
    hybrid = build_hybrid(tiny_cfg, cpu, fp32_policy)
    q = Question("what is the cat", ["sat"], "the cat [sat] down.")
    res = ask(hybrid, [q], cpu, modes=("hybrid",), max_new_tokens=2, log=lambda *a: None)
    with_stem = best_answer_logprob(hybrid.model, hybrid.tokenizer, [qa_prompt(q.text, 0) + " the cat"], [["sat"]], cpu)
    sentence = answer_logprob(hybrid.model, hybrid.tokenizer, [qa_prompt(q.text, 0)], ["the cat sat down."], cpu)
    assert abs(res["modes"]["hybrid"]["fact_logprob"][0] - with_stem[0]) < 1e-5
    assert abs(res["modes"]["hybrid"]["sentence_logprob"][0] - sentence[0]) < 1e-5


def test_fact_recovery():
    assert fact_recovery(-4.0, -1.0, -16.0) == pytest.approx(0.8)
    assert fact_recovery(None, -1.0, -16.0) is None and fact_recovery(-2.0, -1.0, -1.0) is None


QS = ["What is the capital of France?", "How many days are in a week?"]


def _mode(replies, lps, with_refs=True):
    exp = [["Paris"], ["seven", "7"]]
    hits = [_correct(a, e) for a, e in zip(replies, exp)]
    copied = [a == "Grass is green." for a in replies]
    return {"answers": replies, "correct": hits, "fact_logprob": lps, "sentence_logprob": [x - 1 for x in lps],
            "copied": copied, "accuracy": sum(hits) / 2, "mean_fact_logprob": sum(lps) / 2,
            "mean_sentence_logprob": sum(lps) / 2 - 1, "copies": sum(copied), "seconds": 1.0}


def _fake_run(root, name, hybrid, tokens=None, refs=False, scoring=SCORING, questions=QS):
    d = root / name
    d.mkdir()
    res = {"format": "sentence", "scoring": scoring, "raw": False, "max_new_tokens": 40, "questions": questions,
           "expected": [["Paris"], ["seven", "7"]],
           "references": ["The capital of France is [Paris].", "There are [seven] days in a week."],
           "topics": ["geography", "everyday"], "modes": {"hybrid": _mode(*hybrid)}}
    if refs:
        res["modes"]["gemma"] = _mode(["The capital of France is Paris.", "A week has seven days."], [-1.0, -1.0])
        res["modes"]["ablated"] = _mode(["the the the", "of of"], [-16.0, -16.0])
    (d / "qa.json").write_text(json.dumps(res))
    if tokens:
        (d / "summary.json").write_text(json.dumps({"tokens_trained_total": tokens, "final": {"sets": {"mix": {
            "ppl_hybrid": 34.0, "gr_ce": 0.909}}}}))
    return d


def test_compare_three_stages(tmp_path):
    s1 = _fake_run(tmp_path, "stage1_test", (["Grass is green.", "Grass is green."], [-8.0, -5.0]), 983_040, refs=True)
    s2 = _fake_run(tmp_path, "stage2_5m", (["The capital of France is French.", "There are 12 days."], [-4.0, -4.0]),
                   4_997_120, refs=True)
    s3 = _fake_run(tmp_path, "stage3_20m", (["The capital of France is Paris.", "There are 12 days."], [-2.0, -6.0]),
                   20_000_000)
    md = compare([s1, s2, s3])
    assert "# Questions and answers: stage1_test → stage2_5m → stage3_20m" in md
    assert "| No FFN (floor) | – | 0/2 | -16.00 | 0% |" in md
    assert "| Hybrid, stage1_test | 0.98M | 0/2 | -6.50 | 63% | -7.50 | 2/2 |" in md
    assert "| Hybrid, stage2_5m | 5.00M | 0/2 | -4.00 | 80% |" in md
    assert "| Hybrid, stage3_20m | 20.00M | 1/2 | -4.00 | 80% | -5.00 | 0/2 | 34.0 | 0.909 |" in md
    assert "| Gemma (target) | – | 2/2 | -1.00 | 100% |" in md
    assert "| 1 | What is the capital of France? | geography | -1.0 ✓ | -16.0 ✗ | -8.0 ✗ | -4.0 ✗ | -2.0 ✓ | +6.0 |" in md
    assert "the fact got more likely for 1 of 2 questions and less likely for 1" in md
    assert "- stage3_20m: The capital of France is Paris. ✓" in md and "- Gemma: A week has seven days. ✓" in md
    assert "## Replies" not in compare([s1, s2, s3], full=False)


def test_compare_refuses_missing_old_or_mismatched_files(tmp_path):
    new = _fake_run(tmp_path, "stage2_5m", (["Paris.", "seven"], [-2.0, -3.0]))
    with pytest.raises(FileNotFoundError):
        compare([tmp_path / "missing", new])
    old = _fake_run(tmp_path, "stage1_test", (["x", "y"], [-8.0, -5.0]), scoring=None)
    with pytest.raises(ValueError):
        compare([old, new])
    other = _fake_run(tmp_path, "stage3", (["x", "y"], [-8.0, -5.0]), questions=["a?", "b?"])
    with pytest.raises(ValueError):
        compare([new, other])


def test_ask_compare_command(tmp_path):
    from transplant.cli import main

    s1 = _fake_run(tmp_path, "stage1_test", (["Grass is green.", "x"], [-8.0, -5.0]), refs=True)
    s2 = _fake_run(tmp_path, "stage2_5m", (["Paris.", "y"], [-2.0, -6.0]), refs=True)
    out = tmp_path / "cmp.md"
    assert main(["ask-compare", str(s1), str(s2), "--out", str(out)]) == 0
    text = out.read_text()
    assert "stage1_test → stage2_5m" in text and "## Replies" in text
    assert main(["ask-compare", str(tmp_path / "missing"), "--out", str(out)]) == 1
