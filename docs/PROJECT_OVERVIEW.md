# PROJECT_OVERVIEW.md — What This System Is and How It Thinks

<!-- Added 2026-07-23. Onboarding-paced conceptual walkthrough for a new NIC
     maintainer who has never seen this codebase. Precise mechanics live in
     TECHNICAL_REFERENCE_AND_SCALING.md; rules/hard-stops live in AGENTS.md;
     "how do I change X" recipes live in MAINTAINER_PLAYBOOK.md. This file's
     only job is to build the mental model that makes those three legible. -->

## Read this first if you're new

This document explains **why** the system is built the way it is, in the
order a new engineer should learn it. It is deliberately not precise about
hyperparameters, file paths, or exact function signatures — that precision
lives in `TECHNICAL_REFERENCE_AND_SCALING.md` and `MAINTAINER_PLAYBOOK.md`.
Read this first to get the shape of the thing, then go there for the detail.

---

## 1. The problem, and the one non-negotiable constraint

NIC runs a scholarship portal. Some fraction of applications are fraudulent
— fabricated income, colluding "families" sharing a phone number or IP
address to file multiple claims, fee inflation, and similar patterns. The
job of this system is to **rank every application by how suspicious it is**,
so a small human review team can look at the riskiest ones first instead of
sampling randomly or applying manual spot-checks.

**The one constraint that shapes everything else: no rules.** Early versions
of this system (see `HISTORY.md`) used hand-written rules — "flag if income
> X and fee ratio > Y." Rules are exactly the pattern fraud adapts around
once informal knowledge of the thresholds leaks out, and they don't
generalize to fraud patterns nobody has thought of yet. Every version since
has been **rule-free**: numeric thresholds only come from statistics fitted
to the data itself (see §5, EVT), never from a domain expert picking a
number. This is `AGENTS.md` hard stop 1, and it's the reason the whole
architecture looks the way it does — it's built entirely out of things that
can learn "what does normal look like" and flag departures from it,
with no if/else fraud-policy logic anywhere.

## 2. The core idea: an ensemble of "what's unusual" detectors, not a classifier

There usually aren't enough *confirmed* fraud labels to train a normal
supervised classifier well — fraud is rare, confirmations are slow (a human
has to investigate), and a classifier trained on a handful of positives
tends to overfit to coincidental details of those specific cases (this
actually happened — an earlier LightGBM-based fusion layer was destroyed by
just 14 confirmed positives; see `HISTORY.md`).

So instead, the system runs **multiple independent anomaly detectors**, each
looking at the data from a different angle, and each producing a score where
**higher always means more anomalous** (hard stop 3 — this convention is
absolute; if a detector's native output is inverted, it gets flipped at the
point of computation, not left for a human to remember). The detectors don't
need fraud labels to run — they learn what a *typical* application looks
like and score deviation from that.

Three detectors currently feed the final decision:

- **Subspace Isolation Forest** — the backbone. Splits features into
  financial / identity / network groups and runs Isolation Forest per group,
  so an anomaly in income doesn't get diluted by 40 unrelated normal-looking
  features. Works on any application, connected to others or not.
- **Dense-block detector** — a specialist for *collusion rings*: groups of
  applications sharing a mobile number or IP address that are unusually
  densely connected (FRAUDAR-style: repeatedly peel the densest remaining
  subgraph). This catches exactly the failure mode the subspace IF is
  weakest on — a ring of individually-plausible-looking applications that
  are only suspicious *together*.
- **Hybrid GraphMCM** — a graph neural network (RGCN) that learns to predict
  each application's own features from its neighbors' — masking out part of
  the input and asking "given who you're connected to, what should your
  features look like?" A big prediction error means the application doesn't
  fit its own neighborhood, which is a different signal than "this
  application looks weird in isolation."

These three see different failure modes, which is the point — a ring that
looks fine feature-by-feature but is topologically dense gets caught by
dense-block, not subspace IF; an application with no shared identifiers at
all falls back on subspace IF, since it has no graph neighborhood to reason
about.

**A fourth detector — Deep SAD — exists but deliberately isn't part of the
final score.** It was tested directly as a 4th input and didn't improve the
combined result enough to justify the added complexity (see `AGENTS.md` §1)
— the three above already covered its specialty. It still runs, and its
output appears on evidence cards as a supplementary signal a human reviewer
can see, just not something that moves the ranking. This is a useful
precedent to know about before proposing a 5th detector: "does it help
standalone" and "does it help the *fused* score" are different questions,
and this project has already been burned by conflating them once.

## 3. Combining three scores into one: why `max`, not a weighted average

Once you have three anomaly scores per application, how do you combine them
into one ranking? The tempting answer is a weighted sum — give each detector
a coefficient reflecting how much you trust it. **This was tried and
explicitly rejected**, twice:

1. A learned combiner (LightGBM, trained on confirmed labels) got wrecked by
   how few labels existed.
2. A hand-set weighted sum diluted whichever detector had actually found the
   fraud — if dense-block scored a mobile-sharing ring at 0.9 but the other
   two (correctly) saw nothing unusual and scored near 0, summing dragged
   the combined score down toward "unremarkable."

The current fusion is an **unweighted max**: normalize each detector's score
to [0,1] across the population, then take the highest of the three, then
normalize again. This means an application only needs *one* detector to be
confident something's wrong — the other two staying quiet doesn't drag the
score down. It's also inherently label-free, which matters given how little
confirmed-fraud data exists to overfit to.

## 4. The identity graph: what "neighbors" means

Several pieces above (dense-block, the GraphMCM) depend on a notion of which
applications are "connected." The graph is built from **shared identifiers**
— two applications are linked if they share a mobile number, an IP address,
a father's name, a mother's name, or a pincode (5 relation types). This is
deliberately *not* a rule ("flag if 3+ applications share a phone number") —
it's raw structural information that downstream ML components (dense-block,
the GNN) learn to weigh, rather than a threshold anyone hand-picked.

One practical wrinkle worth knowing early: some shared values are extremely
common and not remotely suspicious — a whole town might share a pincode.
Pincode sharing was tried as a dense-block signal and dropped (`AGENTS.md`
§1) precisely because it reflects legitimate geographic clustering, not
collusion, on its own. This is a recurring theme: a signal being *available*
in the graph doesn't mean every detector should use it, and the project's
history has several examples of adding a relation, measuring it made things
worse for a specific fraud category, and pulling it back out.

## 5. Turning scores into a decision: EVT, not a percentile cutoff

A ranked list is useful for a human reviewer, but at some point the system
needs to say "this application clears the bar for automatic promotion to a
pseudo-label" (§6) or "this is the threshold above which review capacity
should focus." Rather than picking a percentile by feel, the system fits an
**Extreme Value Theory (EVT) distribution — a Generalized Pareto Distribution
over the extreme tail** of each score, and derives the threshold from that
fit. This is the *only* place numeric thresholds are allowed to exist (hard
stop 1) — and even here, the threshold comes from a statistical fit to the
observed tail, not a chosen number.

## 6. Self-training: letting confident detections become training signal

Since confirmed fraud labels are scarce, the system has a mechanism to
bootstrap more of them: applications whose scores clear the EVT threshold on
enough independent signals get provisionally treated as likely-fraud
("pseudo-labels") and can feed back into training. This is powerful and
therefore dangerous — if done carelessly, the model could reinforce its own
mistakes in a feedback loop. Two guardrails:

- A pseudo-label requires **agreement across multiple independent EVT
  signals**, not just one (this is itself a tuned threshold — see
  `AGENTS.md` §1 — because requiring only one signal was noisy).
- **Every round of self-training requires a human sign-off** before its
  labels are used (hard stop 5). The system is explicitly coded to never
  auto-advance past round 0. This isn't a suggestion — it's enforced in
  code, and "make self-training fully automatic" is exactly the kind of
  change that should never be made without going back to the project lead.

## 7. Explaining a score: the XAI layer

A risk score alone doesn't help a human reviewer — "why is this flagged?"
matters as much as "how flagged." The XAI layer builds an evidence card per
suspicious application: which features drove its score, which relation
(shared mobile/IP/etc.) connected it to its neighbors and how, and — for the
GraphMCM detector specifically — a post-hoc analysis of *which relation's
removal would have made the model's prediction fit better* (a proxy for
"which connection is driving the suspicion," since the GNN encoder doesn't
have built-in attention weights to read off directly). None of this
narration feeds back into any score — it's presentation only, and it
deliberately never speaks raw identifiers (phone numbers, IPs) directly, to
avoid the XAI layer becoming a de facto rule ("card says mobile-sharing" is
a story a reviewer can act on without a threshold ever being written down as
a rule).

## 8. Everything is checkpointed and swappable, never edited in place

Models retrain periodically (new confirmed labels arrive, self-training
promotes new pseudo-labels, or a scheduled full retrain happens). A new
checkpoint doesn't overwrite the live one directly — it's validated (correct
shape: feature count, embedding dimension, edge-type count must match) at a
temp path, then atomically swapped in (hard stop 9). This exists so that a
bad or incompatible checkpoint can never leave the system in a half-updated,
crashing state — the live model is always either the old one or the fully
validated new one, never something in between.

## 9. The scale migration you're probably arriving mid-way through

The detection logic above (§2–8) is **fixed and validated** — if you're
reading this because something feels off architecturally, the answer is
almost never "change the architecture," it's "check `HISTORY.md` for why it
already looks like this." What *is* actively in motion is the I/O layer:
migrating from CSV/JSON files to PostgreSQL as the system of record, and
from full-graph training to mini-batch (`NeighborLoader`) training, so the
system can run at 3–4 million applications instead of the current 15,000-row
development set. `IMPLEMENTATION.md` has the 5-step plan and its status.
None of §1–8 above change because of this migration — only how the data
gets in and out changes.

---

## Where to go next

| You want to... | Read |
|---|---|
| Know the exact rules you must never break | `AGENTS.md` §4 (hard stops) |
| Understand a component's precise mechanics | `TECHNICAL_REFERENCE_AND_SCALING.md` |
| Make a specific kind of change (add a feature, add a detector, touch the schema) | `MAINTAINER_PLAYBOOK.md` |
| Check what migration step we're on | `IMPLEMENTATION.md` |
| Verify a historical metric claim | `HISTORY.md` (read-only, never extend it) |
| Operate the console / run day-to-day tasks | `OPERATIONS_RUNBOOK.md` |
| Call the API | `API_TESTING_GUIDE.md` |
