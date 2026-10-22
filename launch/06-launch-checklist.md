# DRAFT ONLY, DO NOT POST

## Trainsect launch checklist: what goes live, in what order, once hosting resolves

Two blockers gate everything. Neither is resolved by writing more drafts.

- **Blocker 1, GitHub:** the repo cannot be created by automation (App token returns 403 on repo creation). Habib must create the empty repo in his browser first, then the local repo is pushed.
- **Blocker 2, demo hosting:** Hugging Face now requires PRO for Gradio Spaces, and the free-only rule forbids buying it. The demo Space URL must not be linked anywhere until a free hosting path is confirmed and the page is verified live.

Order of operations once the blockers resolve:

### 1. Habib creates the empty GitHub repo (browser, his action)

- Repo name and visibility per his choice. Nothing else in this checklist can start until this exists.
- Do not create any substitute repo, fork, or mirror under a different account.

### 2. Push the local repo, verify the push

- Push the local `~/workspace/trainsect/` tree to the new remote.
- Verify: README renders, `src/trainsect/` package intact, `demos/`, `docs/`, `tests/`, `validation/` present, LICENSE + CONTRIBUTING.md present.
- Run the test suite from a clean clone: `PYTHONPATH=src python3 -m unittest discover -s tests -t .` Must show 74 tests, all passing, before any launch post goes out.

### 3. Fix the README's Space link BEFORE any launch post

- The README currently links the planned Space URL as a live demo. That URL is not live. Before launch, either replace it with the verified-live demo URL or remove the link and point at the local `trainsect demo` command.
- Never publish a launch post that links a 404. Verify every URL in every draft with a live fetch (expect HTTP 200 + the demo actually loading) before posting.

### 4. Replace every [LINK — goes live after GitHub push] placeholder with the real repo URL

- Files carrying the placeholder: `01-hn-show-hn.md`, `02-x-thread.md`, `03-reddit-ml.md`, `04-linkedin-cards.md`, `05-dev-tutorial.md`.
- LinkedIn file uses the variant `[LINK, goes live after GitHub push]` (no em-dash, per the LinkedIn no-em-dash rule). Replace that variant too.
- After replacement, re-check: zero remaining placeholders across all five drafts, and the repo URL resolves (clone works, README renders).

### 5. Resolve the demo hosting question

- Option A: a free HF Space path exists (confirm in writing, verify the page loads and the demo runs). Only then link it.
- Option B: no free Space path. Then the demo is the terminal recording (see `01-hn-show-hn.md` demo plan) and the local `trainsect demo` command. Do not link the Space URL.
- Do not buy PRO. Do not route around the paywall with a paid tier of any kind.

### 6. Publish in this order

1. **HN Show HN** (`01-hn-show-hn.md`) + terminal recording. First, because it is the canonical announcement the other posts point at.
2. **X thread** (`02-x-thread.md`) + bisection animation. Link the HN post in the final reply if the platform flow allows it.
3. **r/MachineLearning** (`03-reddit-ml.md`). Methods-focused, no cross-posting spam: one post, respond to comments.
4. **LinkedIn cards** (`04-linkedin-cards.md`). Spaced out, one at a time, not all four in one day. Cost-math cards first (cards 1-2), then the magic trick (card 3), then scope (card 4).
5. **DEV tutorial** (`05-dev-tutorial.md`). Last, because it is the evergreen reference the other posts funnel toward.

### 7. Post-launch verification (same day, after each post)

- Each post's repo link resolves. Each post's claims match the demo report (8 replays, 540 steps, 52.2 s, trigger step 60, shard-3, fingerprint 639a024b373ce19a).
- No post links the Space URL unless step 5 option A was verified live.
- Collect failure-case replies (stacks where replay is not bit-exact, predicate misfires, forensic verdicts that should be UNPROVEN). Log them as issues, do not argue them away.

### Never, at any stage

- No posting before the repo is live and the placeholders are replaced.
- No papers, no arXiv, no journal submissions. Drafts only, per the standing rule.
- No paid hosting, no paid promotion, no paid tiers. Free only.
- No touching competitions, hackathons, RSNA, or any other chat's work.
- No inflating numbers. If a re-run on a different machine produces different wall time, quote the new measured number with its machine context, never the old one.
