"""Question-and-answer test: basic questions answered in one full sentence.

Both models are base (pre-trained) language models, not chat models, so each question goes into a
short few-shot frame of three solved questions whose answers are full sentences:

    Answer each question with one complete sentence.

    Q: What is the capital of Italy?
    A: The capital of Italy is Rome.
    ...
    Q: What is the capital of France?
    A:

The model's reply is read greedily up to the end of the line. The three solved examples rotate from
question to question (a pool of six), so a model that only copies the last example cannot score.

Each question has a reference answer with its fact in brackets, "The capital of France is [Paris].",
and one or more key answers ("Paris"). Three models can answer: Gemma (original), the hybrid (FFNs
replaced by the transplanted MoE blocks) and No FFN (Gemma with all transplanted FFNs deleted). For
each model and question:

* **correct**: the reply contains a key answer as a whole word;
* **fact log-prob**: mean log-probability per token of the fact, given the frame, the question and
  the reference sentence up to the fact ("... A: The capital of France is" -> " Paris"); best over the
  key's forms (as written, capitalized, lower-case). This isolates factual recall from phrasing and
  moves long before the replies become right;
* **sentence log-prob**: mean log-probability per token of the whole reference sentence after "A:"
  (phrasing and fact together);
* **copied**: the reply repeats one of the solved examples or the question itself.

``fact recovery`` places the hybrid between No FFN (0%) and Gemma (100%) on the fact log-prob scale,
the question-answering counterpart of GR_CE. ``compare`` lines up the results of several runs
(Stage 1, Stage 2, Stage 3, ...) from their ``qa.json`` files without loading a model.

All questions run as one batch on a fixed-length buffer (right padding, no attention mask: the models
are causal, so padding never reaches the text before it). Few tensor shapes keep the Apple GPU's kernel
cache small.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from ..models import surgery

# Written into qa.json. Results with another value come from an earlier version of the test.
SCORING = "sentence_v1"
RESULT_FILE = "qa.json"
REPORT_FILE = "qa.md"

HEADER = "Answer each question with one complete sentence.\n\n"
EXAMPLES: list[tuple[str, str]] = [  # none of these is a test question
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("How many legs does a spider have?", "A spider has eight legs."),
    ("What color is grass?", "Grass is green."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
    ("What is the largest ocean on Earth?", "The largest ocean on Earth is the Pacific Ocean."),
    ("What do bees make?", "Bees make honey."),
]
N_SHOTS = 3

# (topic, question, reference with the fact in brackets, extra key answers)
DEFAULT_QUESTIONS: list[tuple[str, str, str, list[str]]] = [
    ("geography", "What is the capital of France?", "The capital of France is [Paris].", []),
    ("geography", "What is the capital of Japan?", "The capital of Japan is [Tokyo].", []),
    ("geography", "What is the capital of Germany?", "The capital of Germany is [Berlin].", []),
    ("geography", "In which country are the pyramids of Giza?", "The pyramids of Giza are in [Egypt].", []),
    ("geography", "What language is spoken in Brazil?", "The main language spoken in Brazil is [Portuguese].", []),
    ("geography", "What is the longest river in Africa?", "The longest river in Africa is the [Nile].", []),
    ("geography", "How many continents are there?", "There are [seven] continents.", ["7"]),
    ("science", "What gas do plants absorb from the air?", "Plants absorb [carbon dioxide] from the air.",
     ["CO2", "CO₂"]),
    ("science", "What is the chemical formula of water?", "The chemical formula of water is [H2O].", ["H₂O"]),
    ("science", "At what temperature in Celsius does water freeze?", "Water freezes at [0] degrees Celsius.",
     ["zero"]),
    ("science", "Which planet is known as the Red Planet?", "The Red Planet is [Mars].", []),
    ("science", "What is the largest planet in our solar system?", "The largest planet in our solar system is [Jupiter].",
     []),
    ("science", "What star is at the center of our solar system?", "The star at the center of our solar system is the [Sun].",
     []),
    ("science", "What organ pumps blood through the human body?", "Blood is pumped through the human body by the [heart].",
     []),
    ("science", "What force pulls objects toward the Earth?", "Objects are pulled toward the Earth by [gravity].", []),
    ("math", "What is 2 + 2?", "2 + 2 equals [4].", ["four"]),
    ("math", "What is the square root of 81?", "The square root of 81 is [9].", ["nine"]),
    ("math", "How many sides does a triangle have?", "A triangle has [three] sides.", ["3"]),
    ("math", "What is 10 times 10?", "10 times 10 equals [100].", ["one hundred"]),
    ("everyday", "How many days are in a week?", "There are [seven] days in a week.", ["7"]),
    ("everyday", "How many months are in a year?", "There are [twelve] months in a year.", ["12"]),
    ("everyday", "Which day comes after Monday?", "The day after Monday is [Tuesday].", []),
    ("everyday", "What color is the sky on a clear day?", "On a clear day the sky is [blue].", []),
    ("everyday", "What is the opposite of hot?", "The opposite of hot is [cold].", []),
    ("everyday", "Which animal is known as the king of the jungle?",
     "The animal known as the king of the jungle is the [lion].", []),
    ("everyday", "What is the main ingredient of guacamole?", "The main ingredient of guacamole is [avocado].", []),
    ("history", "Who wrote Romeo and Juliet?", "Romeo and Juliet was written by [William Shakespeare].", ["Shakespeare"]),
    ("history", "Who was the first president of the United States?",
     "The first president of the United States was [George Washington].", ["Washington"]),
    ("history", "Who was the first person to walk on the Moon?", "The first person to walk on the Moon was [Neil Armstrong].",
     ["Armstrong"]),
    ("history", "In which year did World War II end?", "World War II ended in [1945].", []),
]

MODE_LABEL = {"gemma": "Gemma", "hybrid": "Hybrid", "ablated": "No FFN"}
_MODE_OF = {"gemma": "original", "hybrid": "transplant", "ablated": "zero"}


@dataclass
class Question:
    text: str
    answers: list[str] = field(default_factory=list)  # key answers; a reply containing one is correct
    reference: str = ""                                # "The capital of France is [Paris]."; may be empty
    topic: str = ""


def split_reference(reference: str) -> tuple[str, str, str]:
    """'The capital of France is [Paris].' -> ('The capital of France is', 'Paris', '.')."""
    m = re.fullmatch(r"(.*?)\s*\[(.+?)\](.*)", reference.strip())
    if not m:
        return reference.strip(), "", ""
    return m.group(1).strip(), m.group(2).strip(), m.group(3)


def reference_sentence(reference: str) -> str:
    return reference.replace("[", "").replace("]", "").strip()


def default_questions() -> list[Question]:
    out = []
    for topic, q, ref, extra in DEFAULT_QUESTIONS:
        fact = split_reference(ref)[1]
        out.append(Question(q, [fact] + [x for x in extra if x != fact], ref, topic))
    return out


def load_questions(path: Optional[str]) -> list[Question]:
    """The built-in set, or a file with one question per line:

    * ``question | reference with [fact] | alias | ...`` (the fact in brackets is a key answer, aliases too);
    * ``question | answer | alias`` (no reference: only replies and fact log-prob after "A:" are scored);
    * JSON lines with ``question``, optional ``reference``, ``answers`` and ``topic``.
    """
    if not path:
        return default_questions()
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            d = json.loads(line)
            ref = d.get("reference", "")
            keys = list(d.get("answers", []))
            fact = split_reference(ref)[1] if ref else ""
            if fact and fact not in keys:
                keys.insert(0, fact)
            out.append(Question(d["question"], keys, ref, d.get("topic", "")))
            continue
        parts = [p.strip() for p in line.split("|")]
        ref = next((p for p in parts[1:] if "[" in p and "]" in p), "")
        keys = [p for p in parts[1:] if p and p != ref]
        fact = split_reference(ref)[1] if ref else ""
        if fact and fact not in keys:
            keys.insert(0, fact)
        out.append(Question(parts[0], keys, ref))
    return out


def example_indices(k: int) -> list[int]:
    """The solved examples shown before question k: three of the six, rotating with k."""
    return [(k + j) % len(EXAMPLES) for j in range(N_SHOTS)]


def qa_prompt(question: str, k: int = 0) -> str:
    shots = "".join(f"Q: {EXAMPLES[i][0]}\nA: {EXAMPLES[i][1]}\n\n" for i in example_indices(k))
    return f"{HEADER}{shots}Q: {question}\nA:"


def _buffer_length(n: int) -> int:
    return max(64, -(-n // 64) * 64)  # a multiple of 64: few distinct shapes


@torch.no_grad()
def greedy_batch(model, tok, prompts: list[str], max_new_tokens: int, device, stop_at_newline: bool = True) -> list[str]:
    """Greedy continuations of several prompts at once, on one fixed-length buffer."""
    ids = [tok(p, add_special_tokens=True)["input_ids"] for p in prompts]
    length = _buffer_length(max(len(x) for x in ids) + max_new_tokens)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    buf = torch.full((len(ids), length), pad, dtype=torch.long)
    for r, x in enumerate(ids):
        buf[r, : len(x)] = torch.tensor(x)
    buf = buf.to(device)
    cur = [len(x) for x in ids]
    start = list(cur)
    done = [False] * len(ids)
    weight = model.lm_head.weight
    eos = tok.eos_token_id
    rows = torch.arange(len(ids), device=device)
    for _ in range(max_new_tokens):
        h = model.model(input_ids=buf, use_cache=False).last_hidden_state
        last = h[rows, torch.tensor([c - 1 for c in cur], device=device)]
        nxt = (last.to(weight.dtype) @ weight.T).float().argmax(-1).tolist()
        for r, t in enumerate(nxt):
            if done[r]:
                continue
            buf[r, cur[r]] = t
            cur[r] += 1
            text = tok.decode(buf[r, start[r]:cur[r]].tolist(), skip_special_tokens=True)
            if (eos is not None and t == eos) or (stop_at_newline and "\n" in text) or cur[r] >= length:
                done[r] = True
        if all(done):
            break
    outs = []
    for r in range(len(ids)):
        text = tok.decode(buf[r, start[r]:cur[r]].tolist(), skip_special_tokens=True)
        outs.append(text.split("\n")[0].strip() if stop_at_newline else text)
    return outs


@torch.no_grad()
def answer_logprob(model, tok, prompts: list[str], answers: list[str], device) -> list[Optional[float]]:
    """Mean log-probability per token of each answer after its prompt and a space (teacher forcing)."""
    seqs, spans = [], []
    for p, a in zip(prompts, answers):
        if not a:
            seqs.append(tok(p, add_special_tokens=True)["input_ids"])
            spans.append(None)
            continue
        full = tok(p + " " + a, add_special_tokens=True)["input_ids"]
        n_prompt = len(tok(p, add_special_tokens=True)["input_ids"])
        seqs.append(full)
        spans.append((n_prompt, len(full)))
    length = _buffer_length(max(len(s) for s in seqs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    buf = torch.full((len(seqs), length), pad, dtype=torch.long)
    for r, s in enumerate(seqs):
        buf[r, : len(s)] = torch.tensor(s)
    h = model.model(input_ids=buf.to(device), use_cache=False).last_hidden_state
    weight = model.lm_head.weight
    out = []
    for r, span in enumerate(spans):
        if span is None or span[1] <= span[0]:
            out.append(None)
            continue
        lo, hi = span
        logits = (h[r, lo - 1:hi - 1].to(weight.dtype) @ weight.T).float()
        target = buf[r, lo:hi].to(device)
        lp = torch.log_softmax(logits, -1).gather(-1, target[:, None]).squeeze(-1)
        out.append(float(lp.mean()))
    return out


def answer_forms(accepted: list[str]) -> list[str]:
    """Each accepted answer as written, capitalized and lower-case, without repeats."""
    out: list[str] = []
    for a in accepted:
        for form in (a, a[:1].upper() + a[1:], a.lower()):
            if form and form not in out:
                out.append(form)
    return out


def best_answer_logprob(model, tok, prompts: list[str], accepted: list[list[str]], device,
                        batch: int = 32) -> list[Optional[float]]:
    """For each prompt, the best mean log-probability over all forms of its accepted answers."""
    flat_p, flat_a, owner = [], [], []
    for k, (p, acc) in enumerate(zip(prompts, accepted)):
        for form in answer_forms(acc):
            flat_p.append(p)
            flat_a.append(form)
            owner.append(k)
    best: list[Optional[float]] = [None] * len(prompts)
    for start in range(0, len(flat_p), batch):
        lps = answer_logprob(model, tok, flat_p[start:start + batch], flat_a[start:start + batch], device)
        for k, lp in zip(owner[start:start + batch], lps):
            if lp is not None and (best[k] is None or lp > best[k]):
                best[k] = lp
    return best


def _contains_word(text: str, key: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(key.lower()) + r"(?!\w)", text.lower()) is not None


def _correct(reply: str, accepted: list[str]) -> Optional[bool]:
    if not accepted:
        return None
    return any(_contains_word(reply, x) for x in accepted)


def _norm(text: str) -> str:
    return re.sub(r"[^\w ]+", "", text.lower()).strip()


def _copied(reply: str, question: str, k: int) -> bool:
    """The reply repeats one of the solved examples shown with this question, or the question itself."""
    r = _norm(reply)
    if not r:
        return False
    shown = [EXAMPLES[i] for i in example_indices(k)]
    return r == _norm(question) or any(r == _norm(a) or r == _norm(q) for q, a in shown)


def _mean(xs: list) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def ask(hybrid, questions: list[Question], device, modes=("gemma", "hybrid"), max_new_tokens: int = 40,
        raw: bool = False, log=print) -> dict:
    """Replies, hits, fact and sentence log-probabilities for every question and every mode."""
    model, tok = hybrid.model, hybrid.tokenizer
    model.eval()
    prompts = [q.text if raw else qa_prompt(q.text, k) for k, q in enumerate(questions)]
    parts = [split_reference(q.reference) if q.reference else ("", "", "") for q in questions]
    # fact: the reference sentence up to the fact, then the fact (or, with no reference, the key right after "A:")
    fact_prompts = [p + (" " + stem if stem else "") for p, (stem, _, _) in zip(prompts, parts)]
    fact_forms = [([fact] if fact else []) + [a for a in q.answers if a != fact] for q, (_, fact, _) in zip(questions, parts)]
    sentences = [reference_sentence(q.reference) if q.reference else "" for q in questions]
    results: dict = {
        "format": "sentence", "scoring": SCORING, "raw": raw, "max_new_tokens": max_new_tokens,
        "questions": [q.text for q in questions], "expected": [q.answers for q in questions],
        "references": [q.reference for q in questions], "topics": [q.topic for q in questions],
        "modes": {},
    }
    for m in modes:
        t0 = time.time()
        with surgery.mode(model, _MODE_OF[m]):
            replies = greedy_batch(model, tok, prompts, max_new_tokens, device, stop_at_newline=not raw)
            if raw:
                fact_lp = [None] * len(questions)
                sent_lp = [None] * len(questions)
            else:
                fact_lp = best_answer_logprob(model, tok, fact_prompts, fact_forms, device)
                sent_lp = answer_logprob(model, tok, prompts, sentences, device)
        hits = [_correct(a, q.answers) for a, q in zip(replies, questions)]
        copied = [False if raw else _copied(a, q.text, k) for k, (a, q) in enumerate(zip(replies, questions))]
        scored = [h for h in hits if h is not None]
        r = results["modes"][m] = {
            "answers": replies, "correct": hits, "fact_logprob": fact_lp, "sentence_logprob": sent_lp,
            "copied": copied,
            "accuracy": (sum(scored) / len(scored)) if scored else None,
            "mean_fact_logprob": _mean(fact_lp), "mean_sentence_logprob": _mean(sent_lp),
            "copies": sum(copied), "seconds": round(time.time() - t0, 1),
        }
        if r["accuracy"] is None:
            log(f"[ask] {MODE_LABEL[m]}: done ({r['seconds']} s)")
        else:
            log(f"[ask] {MODE_LABEL[m]}: {sum(scored)}/{len(scored)} correct, fact log-prob "
                f"{_fmt(r['mean_fact_logprob'])}, sentence log-prob {_fmt(r['mean_sentence_logprob'])}, "
                f"{r['copies']} copied ({r['seconds']} s)")
    return results


def fact_recovery(hybrid: Optional[float], gemma: Optional[float], no_ffn: Optional[float]) -> Optional[float]:
    """Where the hybrid's fact log-prob sits between No FFN (0) and Gemma (1)."""
    if None in (hybrid, gemma, no_ffn) or gemma == no_ffn:
        return None
    return (hybrid - no_ffn) / (gemma - no_ffn)


def _fmt(x: Optional[float], spec: str = ".2f") -> str:
    return "–" if x is None else format(x, spec)


def _cell(text: str) -> str:
    return (text or "(nothing)").replace("|", "\\|")


def _mark(hit: Optional[bool]) -> str:
    return {True: " ✓", False: " ✗", None: ""}[hit]


def format_markdown(res: dict, label: str = "") -> str:
    """One run's results: a summary per model, every reply, and the fact log-probs as a table."""
    modes = list(res["modes"])
    n = len(res["questions"])
    lines = [f"# Questions and answers {label}".rstrip(), ""]
    if not res["raw"]:
        lines += ["Few-shot frame (base models): three solved questions with one-sentence answers (rotating from a pool "
                  "of six), then the question. Greedy, first line of the reply.", ""]
    head = ["Model", "Correct", "Fact log-prob", "Sentence log-prob", "Copied an example or the question"]
    lines += ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for m in modes:
        r = res["modes"][m]
        scored = [h for h in r["correct"] if h is not None]
        lines.append(f"| {MODE_LABEL[m]} | {sum(scored)}/{len(scored)} | {_fmt(r['mean_fact_logprob'])} | "
                     f"{_fmt(r['mean_sentence_logprob'])} | {r['copies']}/{n} |")
    lines.append("")
    rec = fact_recovery(*(res["modes"].get(m, {}).get("mean_fact_logprob") for m in ("hybrid", "gemma", "ablated")))
    if rec is not None:
        lines += [f"**Fact recovery: {rec:.0%}** (the hybrid's fact log-prob between No FFN = 0% and Gemma = 100%).", ""]

    lines += ["## Replies", ""]
    for k, q in enumerate(res["questions"]):
        ref = reference_sentence(res["references"][k]) if res["references"][k] else ", ".join(res["expected"][k])
        topic = f"{res['topics'][k]}; " if res["topics"][k] else ""
        lines.append(f"**{k + 1}. {q}** ({topic}expected: {ref})")
        lines.append("")
        for m in modes:
            r = res["modes"][m]
            extra = f" · fact {_fmt(r['fact_logprob'][k], '.1f')} · sentence {_fmt(r['sentence_logprob'][k], '.1f')}"
            copy = " · copied" if r["copied"][k] else ""
            lines.append(f"- {MODE_LABEL[m]}: {_cell(r['answers'][k])}{_mark(r['correct'][k])}{extra}{copy}")
        lines.append("")
    lines += ["Fact: mean log-probability per token of the fact after the reference sentence's opening "
              "(\"The capital of France is\" -> \"Paris\"). Sentence: the same for the whole reference sentence. "
              "0 is certain, higher is better.", ""]
    return "\n".join(lines)


def interactive(hybrid, device, modes=("gemma", "hybrid"), max_new_tokens: int = 40, raw: bool = False) -> None:
    """Type a question (or, with raw, any text to continue); empty line or Ctrl-D quits."""
    print("Type a question and press Enter (empty line to quit)." + (" Raw mode: text is continued as is." if raw else ""))
    while True:
        try:
            text = input("\n> ").strip()
        except EOFError:
            break
        if not text:
            break
        res = ask(hybrid, [Question(text)], device, modes=modes, max_new_tokens=max_new_tokens, raw=raw,
                  log=lambda *a: None)
        for m in modes:
            print(f"  {MODE_LABEL[m]:>7}: {res['modes'][m]['answers'][0]}")


# ---- comparing runs (Stage 1, Stage 2, Stage 3, ...) ----------------------------------------------

def _run_facts(run_dir: Path) -> dict:
    """Tokens trained and the final validation-mix numbers of a run, when its files have them."""
    out: dict = {}
    summary = run_dir / "summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        out["tokens"] = s.get("tokens_trained_total") or s.get("e2e_tokens")
        mix = (s.get("final") or {}).get("sets", {}).get("mix")
        if mix:
            out["ppl"], out["gr_ce"] = mix.get("ppl_hybrid"), mix.get("gr_ce")
    return out


def load_run(run_dir) -> dict:
    d = Path(run_dir)
    f = d / RESULT_FILE
    if not f.exists():
        raise FileNotFoundError(f"{f} not found: run `transplant ask --run {d} --ablated` first")
    res = json.loads(f.read_text())
    if res.get("scoring") != SCORING:
        raise ValueError(f"{f} comes from another version of the test ({res.get('scoring')}); run `ask` on it again")
    if "hybrid" not in res["modes"]:
        raise KeyError(f"{f} has no hybrid replies (it has {list(res['modes'])})")
    return {"name": d.name, "res": res, "facts": _run_facts(d)}


def compare(run_dirs: list, full: bool = True) -> str:
    """Markdown: Gemma, No FFN and the hybrid of each run (earliest first) on the same questions.

    Gemma and No FFN do not change from run to run; their replies come from the latest file that has them.
    With ``full``, every reply is listed too.
    """
    runs = [load_run(d) for d in run_dirs]
    base = runs[0]["res"]
    questions = base["questions"]
    for r in runs[1:]:
        if r["res"]["questions"] != questions:
            raise ValueError(f"{r['name']} was asked other questions than {runs[0]['name']}; run `ask` on both again")
    ref_modes = {}
    for m in ("gemma", "ablated"):
        src = next((r["res"]["modes"][m] for r in reversed(runs) if m in r["res"]["modes"]), None)
        if src is not None:
            ref_modes[m] = src
    n = len(questions)
    names = [r["name"] for r in runs]
    lines = ["# Questions and answers: " + " → ".join(names), "",
             "Each question answered in one sentence (few-shot frame, greedy). Gemma is the original model, No FFN is "
             "Gemma with all 26 FFNs deleted, and each run is the hybrid trained in that run.", ""]

    # summary
    g = ref_modes.get("gemma", {}).get("mean_fact_logprob")
    z = ref_modes.get("ablated", {}).get("mean_fact_logprob")
    head = ["Model", "Tokens trained", "Correct", "Fact log-prob", "Fact recovery", "Sentence log-prob", "Copied",
            "Mix perplexity", "GR_CE"]
    lines += ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]

    def row(label, r, facts=None, rec=None):
        scored = [h for h in r["correct"] if h is not None]
        facts = facts or {}
        return ("| " + " | ".join([
            label,
            f"{facts['tokens'] / 1e6:.2f}M" if facts.get("tokens") else "–",
            f"{sum(scored)}/{len(scored)}",
            _fmt(r["mean_fact_logprob"]),
            "–" if rec is None else f"{rec:.0%}",
            _fmt(r["mean_sentence_logprob"]),
            f"{r['copies']}/{n}",
            _fmt(facts.get("ppl"), ".1f"),
            _fmt(facts.get("gr_ce"), ".3f"),
        ]) + " |")

    if "ablated" in ref_modes:
        lines.append(row("No FFN (floor)", ref_modes["ablated"], rec=0.0 if g is not None else None))
    for r in runs:
        h = r["res"]["modes"]["hybrid"]
        lines.append(row(f"Hybrid, {r['name']}", h, r["facts"], fact_recovery(h["mean_fact_logprob"], g, z)))
    if "gemma" in ref_modes:
        lines.append(row("Gemma (target)", ref_modes["gemma"], rec=1.0 if z is not None else None))
    lines.append("")
    if len(runs) > 1:
        a = runs[0]["res"]["modes"]["hybrid"]["fact_logprob"]
        b = runs[-1]["res"]["modes"]["hybrid"]["fact_logprob"]
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        up = sum(1 for x, y in pairs if y - x > 0.05)
        down = sum(1 for x, y in pairs if x - y > 0.05)
        lines += [f"From {names[0]} to {names[-1]}, the fact got more likely for {up} of {len(pairs)} questions and less "
                  f"likely for {down}.", ""]

    # fact log-prob per question
    cols = (["Gemma"] if "gemma" in ref_modes else []) + (["No FFN"] if "ablated" in ref_modes else []) + names
    head = ["#", "Question", "Topic"] + cols + (["Change"] if len(runs) > 1 else [])
    lines += ["## Fact log-prob per question", "", "| " + " | ".join(head) + " |",
              "| " + " | ".join("---" for _ in head) + " |"]
    for k, q in enumerate(questions):
        cells = [str(k + 1), q, base["topics"][k] or "–"]
        for m in ("gemma", "ablated"):
            if m in ref_modes:
                r = ref_modes[m]
                cells.append(f"{_fmt(r['fact_logprob'][k], '.1f')}{_mark(r['correct'][k])}")
        vals = []
        for run in runs:
            h = run["res"]["modes"]["hybrid"]
            vals.append(h["fact_logprob"][k])
            cells.append(f"{_fmt(h['fact_logprob'][k], '.1f')}{_mark(h['correct'][k])}")
        if len(runs) > 1:
            cells.append("–" if None in (vals[0], vals[-1]) else f"{vals[-1] - vals[0]:+.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "✓ / ✗: whether that model's reply contained the answer. Change: last run minus first run.", ""]

    if full:
        lines += ["## Replies", ""]
        for k, q in enumerate(questions):
            ref = reference_sentence(base["references"][k]) if base["references"][k] else ", ".join(base["expected"][k])
            lines += [f"**{k + 1}. {q}** (expected: {ref})", ""]
            if "gemma" in ref_modes:
                r = ref_modes["gemma"]
                lines.append(f"- Gemma: {_cell(r['answers'][k])}{_mark(r['correct'][k])}")
            if "ablated" in ref_modes:
                r = ref_modes["ablated"]
                lines.append(f"- No FFN: {_cell(r['answers'][k])}{_mark(r['correct'][k])}")
            for run in runs:
                h = run["res"]["modes"]["hybrid"]
                lines.append(f"- {run['name']}: {_cell(h['answers'][k])}{_mark(h['correct'][k])}")
            lines.append("")
    lines += ["Fact log-prob: mean log-probability per token of the fact after the reference sentence's opening "
              "(\"The capital of France is\" -> \"Paris\"); 0 is certain, higher is better. Fact recovery: the hybrid's "
              "fact log-prob between No FFN (0%) and Gemma (100%), the question-answering counterpart of GR_CE.", ""]
    return "\n".join(lines)
