# 0001 Land direct to main: no pull requests, one coordinator (`scripts/land-direct`)

Status: accepted (fleet decision via firstmate). Copied from `~/donald-azkg/direct-landing-template/` (vbm-json), adapted. The live plugin checkout is never edited; workers use worktrees.

## Decision
No PRs after the conversion PR. A worker commits locally (never pushes a branch), then runs `scripts/land-direct`, which under one `flock` (`<git common dir>/land.lock`, exit 15 busy) does:
hold check -> red-check -> fetch -> rebase onto `origin/main` in a scratch worktree -> **Tier 1 locally** on that exact combination -> main moved during Tier 1 = rebase and rerun (3 tries) -> hold and red again -> plain `git push origin <sha>:refs/heads/main` (never forced). Each commit lands as itself, linear.
Exit codes: 0 landed, 1 stopped, 2 bad input, 10 red, 11 conflict, 12 baseline worse (unused here), 13 Tier 1 fails, 14 held, 15 busy. Last stdout line is JSON. `--hold "<reason>"` / `--release` use `<common>/hermes-prospecta-land-hold`.

## Settings
- Tier 1 = `scripts/tier1.sh`: the lander self-test plus the full pytest suite with `-n 8` (~40 s local; the suite is 95 s serial, mostly sleeps). Regen = `true` (no generated files).
- Red (the one definition): the `tier1` job of main's latest **push** run of `ci.yml` failed (even while the run goes on). Only a `fix(ci):` / `fix(tests):` / revert commit lands on red. Tier 2 failure never blocks landing.
- Land-time checks: Conventional subject; over 150 production lines (non-test `*.py`) needs a `Diff-Budget:` trailer.
- CI: Tier 1 on push to main only (plus the conversion PR); Tier 2 (serial suite) every 6 h and on dispatch; concurrency group with cancel-in-progress.

## Not ported
The protected-set report, baselines/regen, the Tier 2 revert watchdog (add only once Tier 2 has run on schedule on main), the relabel-attack suite (edits to the lander or Tier 1 script need firstmate main as independent judge). No remote branches exist: `delete_branch_on_merge=true` is the backstop.
