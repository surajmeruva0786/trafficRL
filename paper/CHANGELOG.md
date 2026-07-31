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

## 9. Excess spacing after the biography section removed
**Commit:** `78be508`

- Removed a stray `\vspace{11pt}` inserted before the first author
  biography and a trailing `\vfill` before `\end{document}`, both of
  which produced a visibly large, unprofessional gap between the
  bibliography/biography block and the end of the paper.
- Neither command is used anywhere else in the document — every other
  section header (Introduction, Related Work, Conclusion, etc.) is
  followed immediately by its content with no manual spacing — so
  removing them also brings the biography section in line with the
  formatting convention used throughout the rest of `main.tex`.

## 10. Biography section forced onto a clean column boundary
**Commit:** `bac5d0a`

- After the change above, a rendered PDF still showed an uneven
  biography layout: entries packed into the top of a column followed
  by a large mid-column blank gap before the last entry, because the
  biography block was starting wherever the references happened to
  leave off rather than at a column boundary.
- Added `\newpage` immediately before the first `\begin{IEEEbiographynophoto}`
  so the section always begins at the top of a column instead of
  trying to partially fit into whatever space is left at the bottom of
  the references column. This is the technique the IEEEtran template
  itself documents (`bare_jrnl.tex`: "insert where needed to balance
  the two columns ... with biographies") for this exact situation.
- With three biographies of uneven length (two short, one
  substantially longer for the faculty co-author), this lets the two
  short entries share one column while the long one fills the other
  cleanly, instead of one entry being pushed past a gap.

## 11. Root cause found: \flushbottom stretching the gaps between authors
**Commit:** `48d0aa3`

- The `\newpage` fix above corrected the *starting position* of the
  biography section but not the actual defect: a rendered PDF (page
  13) showed a visibly large, uneven blank gap after Suraj Meruva's
  biography and another after Boini Abhiram's, with Avantika Singh's
  longer entry filling out the rest of the column.
- Root cause: `\documentclass[...,journal]{IEEEtran}` defaults to
  `\flushbottom`, LaTeX's mode for making every column reach exactly
  `\textheight` by stretching the rubber glue between paragraphs. With
  only three short biography entries and nothing else left in the
  column, that glue had a lot of stretching to do, and it was
  distributed as visible gaps between each author rather than as one
  block of unused space at the bottom.
- Fix: added `\raggedbottom` right before the biography section. This
  turns off the page-filling glue stretch for the rest of the
  document, so the three biographies now sit at their natural spacing
  and any leftover column space shows up once, at the bottom of the
  page, which is normal for the last page of an IEEE two-column paper.
- **This turned out to be incomplete** — a subsequent rendered PDF
  still showed the same large gaps between each author (see item 12).
  `\raggedbottom` was a reasonable diagnosis (IEEEtran does default to
  `\flushbottom`) but not the actual mechanism producing the gaps.

## 12. Actual root cause: an infinite-stretch glue hardcoded in IEEEtran.cls
**Commit:** `0baf308`

- Item 11's `\raggedbottom` fix did not remove the gaps in practice, so
  rather than guess again, downloaded IEEEtran.cls itself and read the
  literal definition of the `IEEEbiographynophoto` environment. It
  opens every entry with:
  ```
  \vskip 4\baselineskip plus 1fil minus 0\baselineskip
  ```
- `1fil` is TeX's *infinite*-order stretchable glue, hardcoded directly
  into the class (not a side effect of `\flushbottom`). Whenever a
  column has slack to fill — exactly the situation here, with only
  three short biographies in a column much taller than their combined
  text — TeX always expands the highest-order stretch glue available
  first. With three of these `1fil` skips in the same column (one per
  biography), all of the slack gets distributed across them as a
  visible gap before every entry, regardless of `\raggedbottom` or
  `\flushbottom`, which is exactly why item 11's fix had no visible
  effect.
- IEEEtran does not expose a public hook to change just that one skip,
  so `IEEEbiographynophoto` is locally redefined in `main.tex`
  (wrapped in `\makeatletter`/`\makeatother`) with a body identical to
  the original except the elastic `\vskip 4\baselineskip plus 1fil
  minus 0\baselineskip` is replaced with a small fixed, non-stretchable
  `\vskip 1.5\baselineskip`. Every other internal macro the environment
  calls (`\@IEEEcompsoconly`, `\@IEEEgobbleleadPARNLSP`,
  `\@IEEEbiographyTOCentrynotmade`, the `IEEEbiography` counter) was
  checked against the upstream class source to confirm the override
  reproduces the original behavior exactly, minus the stretch.

## 13. Figures 1-3 replaced with generated architecture diagrams
**Commit:** `862b6ed`

- Added three externally generated images — `fig1RL.png` (traffic
  regime classification), `fig2RL.png` (generic MH-DQN system
  overview), `fig3RL.png` (detailed MH-DQN architecture, including
  per-head replay buffer subsets and target-network sync) — and swapped
  them in for the hand-drawn TikZ versions of the same three figures
  that previously lived directly in `main.tex`.
- Verified each image's content against the section it illustrates
  before wiring it in: Figure 1 matches the Low/Medium/High regime
  description in the Introduction (with the numeric queue/wait
  thresholds deliberately omitted from the image, since those are
  already given in prose in Section III); Figure 2 matches the
  system-overview description in Section III-C; Figure 3 matches the
  MH-DQN architecture description and, in its added replay-subset and
  target-sync detail, the Training Objective subsection (Eq. 3 and the
  100-step target-sync interval already described in the text).
- Figures 2 and 3 were promoted from single-column `figure` floats to
  full-width `figure*` floats (`width=6.8in`, matching the convention
  already used elsewhere for dense images such as `training_curves.png`)
  since the new images carry more detail — legends, per-head math
  notation — than a single 3.4in column renders legibly.
- Figure 3's caption was extended by one clause to mention the replay
  subset $\mathcal{D}_k$ and 100-step target-network sync now shown
  explicitly in the image; both details were already present in the
  paper's running text, so this is a caption clarification, not a new
  claim.
- Removed `\usepackage{tikz}` and `\usetikzlibrary{...}` from the
  preamble, since after this change no figure in the document draws
  with TikZ anymore.
