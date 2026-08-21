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

## 14. Figure 1 caption shortened
**Commit:** `ce2a7a8`

- Trimmed the Figure 1 caption down to a single sentence: "Three
  operational regimes at a signalized intersection. Each regime is
  characterized by a distinct queue length and waiting time range."
- Removed the trailing clauses about each regime requiring a different
  control strategy and about a single monolithic policy motivating the
  multi-head architecture — that reasoning is still made in the
  surrounding Introduction prose, so nothing is lost, only de-duplicated
  out of the caption.

## 15. Related Work refreshed with 2022-2026 literature
**Commit:** `903f05d`

- The RL-TSC portion of Section II and Table 1 previously cited only
  2015-2020 work (Mnih 2015 DQN, IntelliLight 2018, Double/Dueling DQN
  2016, Chu 2020 multi-agent RL, Mannion 2018 transfer learning, Jacobs
  1991 mixture-of-experts, Haydari 2022 survey). Replaced these with
  eight verified papers from 2022-2026, sourced via live web search and
  cross-checked against arXiv/publisher listings for correct authors,
  venues, and years before citing:
  - Chen, Fang, and Sadeh, "The Real Deal" (arXiv:2206.11996, 2022) —
    review of RL-TSC field-deployment barriers.
  - Mei, Lei, Da, Shi, and Wei, "LibSignal" (arXiv:2211.10649, 2022;
    *Machine Learning*, 2023) — open cross-simulator RL-TSC benchmark.
  - Zhao, Dong, Cao, and Chen, DRL-TSC survey (*Eng. Appl. Artif.
    Intell.*, vol. 133, art. 108100, 2024).
  - Jiang et al., "X-Light" (IJCAI 2024) — transformer-on-transformer
    meta-RL for cross-city transfer.
  - Yao, Sun, Lu, Wang, and Yu, mixture-of-experts SAC for connected/
    automated-vehicle highway decisions (*Chin. J. Mech. Eng.*, vol. 38,
    no. 1, 2025) — used in place of Jacobs 1991 to make the
    specialized-sub-policy argument with a recent, traffic-adjacent RL
    result instead of the 1991 foundational MoE paper.
  - Yuan, Lai, and Liu, "CoLLMLight" (arXiv:2503.11739, 2025; ICLR 2026
    poster) — cooperative LLM agents for network-wide TSC.
  - Zhang, Nassir, Chan, and Haghani, "MA2B-DDQN" (arXiv:2602.02959,
    2026) — equity-aware action-branching double DQN.
  - Xiao et al., RL-TSC survey (*Artif. Intell. Rev.*, vol. 59, no. 5,
    2026).
  Two of the eight (Zhang et al. and Xiao et al.) are 2026 publications,
  satisfying the request to visibly relate this work to the most
  current literature.
- Rewrote the "Reinforcement Learning for Traffic Signal Control"
  subsection's three paragraphs around these works and rebuilt Table 1's
  eight rows (Author/Year, Approach, Limitation) so each entry's stated
  limitation is the one this paper's regime-aware multi-head
  architecture actually addresses, rather than reusing the old
  algorithm-history framing (Double DQN, Dueling DQN, PER as isolated
  algorithmic refinements).
  Also updated two citations in the Introduction (the general
  congestion-cost claim and the "single policy underperforms a
  specialized one" claim) to point at the new survey and
  mixture-of-experts references instead of the retired 2018/2020
  citations.
- Left the classical ATSC subsection (SCOOT, SCATS, model predictive
  control) and citations used elsewhere in the paper for algorithmic or
  tooling foundations — Mnih 2015 (DQN, still needed for the Training
  Objective section), Sutton & Barto, Watkins Q-learning, Adam, PyTorch,
  SUMO, and CoLight (cited in Future Work) — untouched, since those are
  method/tooling foundations rather than competing TSC approaches being
  surveyed in Table 1.
- One stray citation of the removed `liang2019deep` key in the classical
  ATSC paragraph (unrelated to the RL subsection, about MPC models being
  hard to maintain in the field) was repointed to Chen et al. 2022,
  which makes the same practical-deployment argument. Verified with a
  scripted check that every `\cite{}` key in `main.tex` now resolves to
  a defined `\bibitem{}` and vice versa, and that all table/figure/
  bibliography LaTeX environments remain balanced.
- Net bibliography size: 22 entries before this change, 18 after (12
  outdated entries removed, 8 added).

## 16. New 2022-2026 literature woven into Discussion and Future Work
**Commit:** `5024d6f`

- The previous entry (\#15) only touched Section II. This pass carried
  the same eight references into the parts of the paper where the
  paper compares itself against, or points toward, prior/related
  approaches, so the refresh isn't confined to one section:
  - **Discussion**, "Decentralized multi-agent control is effective"
    paragraph: added a closing comparison contrasting this work's
    implicit, communication-free coordination with CoLLMLight's
    ~\cite{yuan2025collmlight} per-step LLM inference across cooperating
    agents and X-Light's~\cite{jiang2024xlight} single shared
    cross-city policy — positioning the decentralized MH-DQN result
    against two concrete recent alternatives instead of an unspecific
    claim.
  - **Future Work**, all six bullets now cite the specific recent work
    each proposed extension would build on:
    1. *Reward Function Tuning* — added a pointer to equity-aware
       reward terms, citing Zhang et al.~\cite{zhang2026equity}.
    2. *Explicit Multi-Agent Coordination* — added
       \cite{yuan2025collmlight} (cooperating LLM agents) alongside the
       existing CoLight~\cite{wei2019colight} citation.
    3. *Hierarchical Policy Learning* — left unchanged; no 2022-2026
       reference in the working set matched this bullet specifically.
    4. *Dynamic Regime Threshold Adaptation* — added
       \cite{yao2025moesac}, since learned MoE gating is the closest
       existing analogue to adaptive regime-boundary learning.
    5. *Real-World Deployment* — added \cite{chen2022realdeal} (the
       barriers this bullet describes are the ones that paper
       enumerates) and \cite{mei2023libsignal} (LibSignal) as a
       cross-simulator validation step to precede field deployment.
    6. *Extended Corridor Networks* — added \cite{jiang2024xlight}
       (cross-city transfer) and \cite{zhang2026equity}
       (corridor-to-network action-branching) as two concrete scaling
       strategies.
- Zhao et al. 2024 and Xiao et al. 2026 (the two DRL-TSC surveys) were
  left cited only in the Introduction and Related Work, where survey
  papers belong; they don't describe a specific technique for the
  Discussion or Future Work to build on or contrast against.
- Re-ran the same scripted check as entry \#15 after this pass: every
  `\cite{}` key resolves to a defined `\bibitem{}` and vice versa (18
  defined, 18 used, zero orphans in either direction), and all
  table/figure/bibliography LaTeX environments remain balanced.

## 17. Figures 2, 6, 7, and 11 removed
**Commit:** `49f02e8`

- Removed four of the paper's 18 figures (14 remain), and rewrote every
  paragraph that depended on one of them so nothing was left describing
  a plot that no longer exists:
  - **Figure 2** (`fig:system_overview`, `fig2RL.png`) — the generic
    MH-DQN system-architecture diagram. It largely duplicated the
    detailed architecture figure (`fig3RL.png`, now Figure 2) a few
    paragraphs later, so the sentence in "System Architecture and
    Methodology" that pointed to it was rewritten into an inline prose
    description of the same state → shared trunk → classifier →
    Q-heads → soft gating → action pipeline, with a forward pointer to
    the detailed figure instead.
  - **Figure 6** (`fig:episodes`) and **Figure 7** (`fig:distribution`)
    — a paired per-episode line plot and box-plot distribution for
    Stage 2, both used in one paragraph to argue that the reported
    averages aren't hiding a few bad episodes. Rewrote that paragraph
    as prose grounded in Table II (`tab:results_sim`): the MH-DQN still
    beats the fixed-time baseline in all 10 evaluation episodes with no
    overlap, the claim just isn't illustrated with a plot anymore.
  - **Figure 11** (`fig:training`) — the Stage 2 training-dynamics plot
    (episode reward, Q-loss, classifier loss/accuracy over 200
    episodes). It was central to two paragraphs. The first explained an
    apparent contradiction between near-chance training-time classifier
    accuracy and the 100% evaluation accuracy reported earlier; that
    paragraph now cites only the two confusion-matrix figures that
    remain (`fig:confusion_train`, `fig:confusion`) and describes the
    accuracy-curve and classifier-loss numbers as findings from the
    underlying training logs rather than a figure. The second paragraph
    (Q-loss convergence, reward drift, waiting-time stability) is now
    framed the same way, as a description of the logs rather than "the
    remaining panels of Fig. X."
- Renumbered the `% FIGURE N` comment headers for all 14 remaining
  figures so they stay sequential (1-14); these are source comments
  only and don't affect the compiled output, but were kept accurate for
  anyone editing the file later. Also deleted one now-orphaned comment
  separator left behind by the Figure 6/7 removal.
- The four now-unreferenced image files (`fig2RL.png` and three of the
  `WhatsApp Image ...jpeg` files) were left in the repository rather
  than deleted, since removing files wasn't part of what was asked;
  they're simply no longer pulled into the PDF by any
  `\includegraphics`.
- Verified with the same scripted checks used in entries \#15-16: every
  `\ref{fig:...}` resolves to a defined `\label{fig:...}` and vice versa
  (14 defined, 14 used, zero orphans), every `\cite{}`/`\bibitem{}`
  pairing is still intact and untouched by this change, and all
  figure/table/bibliography LaTeX environments remain balanced
  (14 `\begin{figure}` / `\end{figure}` pairs, matching the 14 labels).

## 18. Original Figures 8 and 18 removed, plus original Table 2
**Commits:** `e67b1b8`, `8b550ba`

- This request used figure/table numbers rather than labels, and by
  this point in the session the paper's own figure numbering had
  already shifted twice (18 figures originally, then 14 after entry
  \#17). Asked the user to confirm which numbering they meant before
  touching anything, since "Figure 18" no longer existed under the
  current (post-\#17) numbering; they confirmed the *original* 1-18
  numbering, from before any figure was ever removed this session —
  i.e., the last point at which both Figure 8 and Figure 18 existed
  together.
- **Original Figure 8** (`fig:regime`, "Traffic regime distribution
  during training", Stage 2) removed. The Stage 2 results paragraph
  had framed this as the first of "three further figures" explaining
  why the MH-DQN reaches its result; rewrote it to state the
  32%/41.8%/26.2% low/medium/high training-time split as a plain
  finding from the training logs (no figure reference), and
  renumbered the remaining two points — classifier accuracy
  (`fig:confusion`) and head specialization (`fig:specialization`) —
  from Second/Third to First/Second.
- **Original Figure 18** (`fig:episodes_hab`, "Per-episode comparison
  over 5 evaluation episodes", Stage 5 corridor) removed. The sentence
  that pointed to it now states the same finding as prose grounded in
  Table VIII (`tab:results_hab`): the MH-DQN beats the fixed-time
  baseline on waiting time and queue length in all 5 evaluation
  episodes, with no reversals.
- **Original Table 2** (`tab:stages`, "Summary of Five Development
  Stages") removed, per a mid-turn follow-up request specifying the
  original table numbering (before any removals — this was the first
  table ever removed this session, so original and current numbering
  still coincided). Folded its five rows (stage name, environment,
  episode count) into the lead sentence of "Overall Development
  Pipeline" as prose; the following "Episode budget rationale"
  paragraph already explained the 200-vs-100 episode split and needed
  no change.
- Renumbered the `% FIGURE N` source comments for all 12 remaining
  figures (down from 14) to stay sequential.
- Two more image files became unreferenced by the figure removals
  (`WhatsApp Image 2026-02-10 at 9.05.32 AM.jpeg`, the Figure 8
  source, and `per_episode_comparison.png`, the Figure 18 source) and
  were left in the repository for the same reason as entry \#17's
  four files: deleting files wasn't part of what was asked.
- Verified with the same scripted checks as entries \#15-17: every
  `\ref{fig:...}`/`\label{fig:...}` pair resolves in both directions
  (12 defined, 12 used, no orphans), every `\ref{tab:...}`/
  `\label{tab:...}` pair resolves in both directions (11 tables
  remain, down from 12, no orphans), citations are untouched (18
  defined, 18 used), and all figure/table/bibliography LaTeX
  environments remain balanced.
