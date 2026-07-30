# Paper Changelog

Summary of all revisions made to `main.tex` (the paper draft) in the current
editing pass. Ordered chronologically; each entry corresponds to one commit
on `main`. No numeric results, table values, or formulas were changed at any
point in this process — every edit below is either prose, notation
presentation, metadata, or a technical/formatting fix.

## 1. Language and readability rewrite
**Commits:** `4c120ac`, `654c7cb`

- Rewrote the entire paper's prose for coherence and simplicity: abstract,
  introduction, related work, methodology, all five stage sections,
  discussion, future work, and conclusion.
- First pass made the language too casual (news-article tone); second pass
  restored a formal, professional scientific register while keeping
  sentence structure simple and the stage-by-stage narrative easy to follow.

## 2. Mathematical notation formalized
**Commit:** `d75615e`

- Added a **Notation** subsection with a symbol-reference table covering
  every variable used in the MDP formulation, reward function, and
  soft-gating equation.
- Rewrote the reward function with precise, well-defined symbols (normalized
  waiting time, queue fraction, throughput count, phase-switch indicator)
  in place of bare words, verified against the actual reward computation in
  `traffic_rl/env/sumo_env.py`.
- Added explicit variable definitions to the soft-gating and TD-loss
  equations, introduced `K=3` and the regime-subset replay notation
  `D_k`, and clarified which parts of the formulation are standard DQN vs.
  specific to this work's architecture.

## 3. Author list, affiliations, and biography
**Commits:** `2d0c8a6`, `29862c2`, `2a4c93b`

- Added Avantika Singh as a third author (Senior Member, IEEE), listed
  under the Department of DSAI alongside Suraj Meruva and Boini Abhiram.
- Replaced the incorrect "Student Member, IEEE" designation for the two
  student authors with their department affiliation.
- Updated the Acknowledgment to thank IIIT Naya Raipur and Avantika Singh.
- Added a full biography entry for Avantika Singh (education, prior
  appointments, current role, research interests).

## 4. Genuine result analysis and figure explanations
**Commit:** `dea5584`

- Added honest, plain-language analysis to every Results subsection,
  including two findings surfaced by directly inspecting the code and the
  actual figure images rather than assuming:
  - Explained why the MH-DQN's raw reward is *lower* than the fixed-time
    baseline's despite better real-world outcomes (throughput carries a
    10x larger reward coefficient than wait/queue).
  - Traced a training-vs-evaluation classifier-accuracy discrepancy to its
    root cause in `multihead_agent.py` (both training-time metrics are
    cumulative averages over the entire history, so early exploration
    noise never washes out).
  - Corrected an inaccurate claim about head-specialization scores
    "increasing steadily" after checking the actual plot (it fluctuates
    with two sharp peaks).
  - Added a block-by-block reading of the Stage 5 fine-tuning run showing
    reward crashes early as agents unlearn a weaker phase-change penalty
    before recovering.
- Added inline references and explanations for all 18 figures in the
  paper (8 previously had labels but were never cited in the text).

## 5. Empirical basis of regime thresholds clarified
**Commit:** `918357f`

- Added a sentence after the Low/Medium/High regime threshold definitions
  stating these values were obtained through iterative experimentation
  while tuning the model, not adopted from prior published work.

## 6. Corridor selection rationale
**Commit:** `742975e`

- Added two sentences explaining why the Habsiguda–Nacharam corridor was
  chosen: it is a well-known high-congestion arterial route in Hyderabad,
  and the authors' local familiarity with the area supported accurate
  construction of the SUMO network.

## 7. Technical/formatting fixes
**Commits:** `e0047c3`, `6b07701`, `62cbdbd`

- Renamed `stage1&2.png` → `stage1_2.png` and `stage4&5.png` →
  `stage4_5.png`. An unescaped `&` is a special LaTeX character that can
  break `\includegraphics`.
- Fixed a genuine broken image reference: the file was actually named
  `habsiguda_map.png.jpeg` (and is a true JPEG by content) while the code
  referenced a nonexistent `habsiguda_map.png`. Renamed the file to
  `habsiguda_map.jpg` and updated the reference — this was the cause of
  the empty figure in the Stage 4/5 section.
- Changed `\usepackage{hyperref}` to `\usepackage[hidelinks]{hyperref}` to
  remove the default colored PDF link borders (red around figure/table
  references, green around citations) that were being mistaken for
  compile errors.
- Audited the full document for duplicate labels, broken `\ref`/`\cite`
  targets, table column/row consistency, caption placement vs. IEEE
  style, and breakable spaces before references/units: all clean.

## 8. Bibliography expanded from 12 to 22 references
**Commit:** `6435c4d`

- Added 10 new references, each verified via web search for accurate
  bibliographic details, and cited every one in the running text (not
  left in the bibliography alone):
  - Sutton & Barto, *Reinforcement Learning: An Introduction* (2018) —
    cited at the MDP formulation.
  - Watkins & Dayan, "Q-learning" (1992) — cited with the TD-learning
    objective.
  - Hunt et al., the original SCOOT paper (1982), and Sims & Dobinson,
    the original SCATS paper (1980) — these fill in two literal empty
    `[]` citation placeholders that existed in the Related Work section.
  - Varaiya, "Max pressure control..." (2013) — cited alongside the
    already-present PressLight reference.
  - El-Tantawy et al., MARLIN-ATSC (2013) — cited alongside Chu et al.
  - Kingma & Ba, Adam (2015) — cited where training uses the Adam
    optimizer.
  - Paszke et al., PyTorch (2019) — cited in the Acknowledgment.
  - Wei et al., CoLight (2019) — cited in the Future Work item on
    GNN/attention-based coordination.
  - Krajzewicz et al., SUMO (2012) — cited alongside Lopez et al.
- Verified every `\cite` key resolves to a `\bibitem` and every
  `\bibitem` is cited at least once (no orphans in either direction).
