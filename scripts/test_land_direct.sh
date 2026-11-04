#!/usr/bin/env bash
# Tier 1: scripts/land-direct (docs/decisions/0001-land-direct.md) can land and can stop. Against a local bare remote (denyNonFastForwards, and an update hook that records every ref update and can
# refuse pushes), a gh shim for the red-check, a stand-in Tier 1 ("fails when BAD exists", with a per-call hook) and a stand-in regenerator:
#  a commit lands on main as itself (no branch, no PR, linear, one JSON line); several land in order; two workers racing: one waits for the lock and lands after, a third that cannot get it exits 15 and lands
#  nothing; a conflicting rebase aborts naming the file; a baseline conflict regenerates by itself, a worsening one stops; a failing Tier 1 lands nothing; main moving during Tier 1 re-runs; a refused push is
#  retried (3 tries) and never forced; the hold marker stops it (also one set during Tier 1); a red main blocks all but a fix(ci): commit; bad subject / >150 production lines stop; bad input is exit 2.
#  Negative self-test: the no-force scan is shown to catch a force push in a fixture.
export LC_ALL=C
set -euo pipefail
cd "$(dirname "$0")/.."
land=$(pwd)/scripts/land-direct
fail() { echo "tier1: land-direct: $*" >&2; exit 1; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; export T
noforce() { grep -v '^[[:space:]]*#' "$1" | grep -E -- '--force|--mirror|--delete|push[^|;&]* -f |push[^|;&]* \+|push[^|;&]* :' || true; }
[ -z "$(noforce "$land")" ] || fail "scripts/land-direct contains a force/delete push: $(noforce "$land")"
printf 'git push --force-with-lease origin x\n' > "$T/forced.sh"; [ -n "$(noforce "$T/forced.sh")" ] || fail "negative self-test: the no-force scan missed a force push"
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t
git init -q --bare -b main "$T/remote.git"; git -C "$T/remote.git" config receive.denyNonFastForwards true
cat > "$T/remote.git/hooks/update" <<'SH'
#!/bin/sh
echo "$1 $2 $3" >> "$T/updates"
if [ "$1" = refs/heads/main ] && [ "$(cat "$T/reject" 2>/dev/null || echo 0)" -gt 0 ]; then echo $(( $(cat "$T/reject") - 1 )) > "$T/reject"; echo "main moved: refused by the test" >&2; exit 1; fi
SH
chmod +x "$T/remote.git/hooks/update"
git clone -q "$T/remote.git" "$T/work" 2>/dev/null; cd "$T/work"; git config core.hooksPath /dev/null
git clone -q "$T/remote.git" "$T/other" 2>/dev/null; git -C "$T/other" config core.hooksPath /dev/null
cat > "$T/regen.sh" <<'SH'
for m in m1 m2 m3; do cur=$(cat $m); old=$(awk -v k=$m '$1==k{print $2}' base.txt); [ "$cur" -le "$old" ] || exit 3; done
for m in m1 m2 m3; do echo "$m $(cat $m)"; done > base.txt
SH
cat > "$T/tier1.sh" <<'SH'
test ! -e BAD || { echo "BAD exists" >&2; exit 1; }
n=$(cat $T/t1n 2>/dev/null || echo 0); n=$((n + 1)); echo $n > $T/t1n
[ ! -e $T/hook.$n ] || sh $T/hook.$n
echo "closure gate: spec 4 | boundary 0 | unverified 1 | total 5 | specified share 80.00% (floor 80.00%)"
SH
cat > "$T/gh" <<'SH'
#!/bin/bash
case "$1 $2" in
  "run list") cat $T/latest 2>/dev/null || true;;
  "run view") cat $T/jobs 2>/dev/null || true;;
esac
SH
chmod +x "$T/gh"
export GH="$T/gh" TIER1_CMD="sh $T/tier1.sh" REGEN_CMD="sh $T/regen.sh" LAND_LOCK_WAIT_SECONDS=60
cm() { git add -A; git commit -qm "${2:-feat(test): $1}"; }
cmo() { git -C "$T/other" add -A; git -C "$T/other" commit -qm "feat(test): $1"; }
printf '10\n' > m1; printf '10\n' > m2; printf '10\n' > m3; printf 'm1 10\nm2 10\nm3 10\n' > base.txt; printf 'line1\nline2\nline3\n' > code.txt
mkdir -p scripts; printf '# generated\nbase.txt\n' > scripts/generated-paths.txt
cm base; git push -q origin HEAD:main; git -C "$T/other" pull -q origin main
git worktree add -q -b w2 "$T/w2" main; git worktree add -q -b w3 "$T/w3" main   # more workers on the SAME git common dir
common=$(git rev-parse --path-format=absolute --git-common-dir)
sync() { git fetch -q origin; git checkout -q -B "$1" origin/main; }
run() { "$land" "$@" >"$T/out" 2>&1 && rc=0 || rc=$?; }
jf() { python3 -c 'import json,sys; print(json.loads(sys.stdin.read().splitlines()[-1])[sys.argv[1]])' "$1" < "$T/out"; }
jfile() { python3 -c 'import json,sys; print(json.loads(open(sys.argv[2]).read().splitlines()[-1])[sys.argv[1]])' "$1" "$2"; }
main() { git ls-remote origin refs/heads/main | cut -f1; }
rgit() { git --git-dir="$T/remote.git" "$@"; }
showf() { rgit show "main:$1"; }
commit_in() { echo "$2" > "$1/$2.txt"; git -C "$1" add -A; git -C "$1" commit -qm "feat(test): $2"; }
# 1. a commit lands as itself, direct to main
sync a; echo a > a.txt; cm a; head0=$(git rev-parse HEAD); : > "$T/updates"
run; [ "$rc" = 0 ] && [ "$(jf result)" = landed ] && [ "$(jf sha)" = "$(main)" ] && [ "$(jf commits)" = 1 ] && [[ "$(jf tier1_seconds)" =~ ^[0-9]+$ ]] || fail "a did not land: $(cat "$T/out")"
[ "$(rgit log -1 --format=%s main)" = "feat(test): a" ] || fail "a: the commit did not land as itself"
[ "$(wc -l < "$T/updates")" = 1 ] && [ "$(rgit for-each-ref --format='%(refname)')" = refs/heads/main ] || fail "a: the push touched more than main (no branch, no tag): $(cat "$T/updates")"
[ "$(git rev-parse HEAD)" = "$head0" ] && [ -z "$(git status --porcelain)" ] || fail "a: the worker's own worktree was touched"
# 1b. a dry run pushes nothing
sync d1; echo d1 > d1.txt; cm d1; m0=$(main); run --dry-run; [ "$rc" = 0 ] && [ "$(jf result)" = would-land ] && [ "$(main)" = "$m0" ] || fail "dry run: $(cat "$T/out")"
# 2. several local commits land in order, linear
sync b; echo b1 > b1.txt; cm b1; echo b2 > b2.txt; cm b2; run; [ "$rc" = 0 ] && [ "$(jf commits)" = 2 ] || fail "b did not land: $(cat "$T/out")"
[ "$(rgit log -2 --reverse --format=%s main | paste -sd,)" = "feat(test): b1,feat(test): b2" ] && [ -z "$(rgit rev-list --merges main)" ] || fail "b: not two linear commits in order"
# 3. race: A holds the lock in its Tier 1; C (1 s patience) is told busy and lands nothing; B waits and lands after A
sync r1; echo r1 > r1.txt; cm r1; git -C "$T/w2" checkout -q -B r2 origin/main; commit_in "$T/w2" r2; git -C "$T/w3" checkout -q -B r3 origin/main; commit_in "$T/w3" r3
rm -f "$T/t1n"; printf 'touch $T/a.started\nwhile [ ! -e $T/a.go ]; do sleep 0.05; done\n' > "$T/hook.1"; m0=$(main)
timeout 120 "$land" > "$T/outA" 2>&1 & pa=$!
for _ in $(seq 400); do [ -e "$T/a.started" ] && break; sleep 0.05; done; [ -e "$T/a.started" ] || fail "race: A never reached its Tier 1"
( cd "$T/w2" && LAND_LOCK_WAIT_SECONDS=1 timeout 60 "$land" ) > "$T/outC" 2>&1 && rcC=0 || rcC=$?
[ "$rcC" = 15 ] && [ "$(jfile result "$T/outC")" = busy ] && [ "$(main)" = "$m0" ] || fail "race: a worker that cannot get the lock is not told busy (exit $rcC): $(cat "$T/outC")"
( cd "$T/w3" && timeout 120 "$land" ) > "$T/outB" 2>&1 & pb=$!
for _ in $(seq 400); do grep -q "waiting for the lock" "$T/outB" 2>/dev/null && break; sleep 0.05; done
[ "$(main)" = "$m0" ] || fail "race: main moved while A still held the lock"
touch "$T/a.go"; wait $pa && wait $pb || fail "race: A or B failed: $(cat "$T/outA" "$T/outB")"
[ "$(rgit log -2 --reverse --format=%s main | paste -sd,)" = "feat(test): r1,feat(test): r3" ] && [ "$(cat "$T/t1n")" = 2 ] || fail "race: not A then B (tier1 x$(cat "$T/t1n")): $(rgit log --format=%s -4 main)"
rm -f "$T/hook.1"
# 4. a conflicting rebase aborts naming the file; main untouched; no rebase left in the lander's worktree
sync cf; sed -i 2s/line2/mine/ code.txt; cm cf; git -C "$T/other" pull -q origin main; sed -i 2s/line2/theirs/ "$T/other/code.txt"; cmo mv; git -C "$T/other" push -q origin HEAD:main; m0=$(main)
run; [ "$rc" = 11 ] && [ "$(jf result)" = conflict ] && grep -q "not generated: code.txt" "$T/out" && [ "$(main)" = "$m0" ] || fail "a code conflict was not stopped (exit $rc): $(cat "$T/out")"
lw=$common/hermes-prospecta-land-direct/wt
[ -z "$(git -C "$lw" status --porcelain)" ] && [ ! -d "$(git -C "$lw" rev-parse --path-format=absolute --git-path rebase-merge)" ] || fail "the lander worktree is left dirty or in a rebase"
# 5. a baseline both sides rewrote lands by itself; a worsening one stops (12)
sync g1; echo 8 > m1; printf 'm1 8\nm2 10\nm3 10\n' > base.txt; cm g1
git -C "$T/other" pull -q origin main; echo 7 > "$T/other/m2"; printf 'm1 10\nm2 7\nm3 10\n' > "$T/other/base.txt"; cmo g2; git -C "$T/other" push -q origin HEAD:main
run; [ "$rc" = 0 ] && grep -q "took main's and regenerating" "$T/out" && [ "$(showf base.txt)" = "$(printf 'm1 8\nm2 7\nm3 10')" ] || fail "g1: a baseline conflict did not resolve by itself (exit $rc): $(cat "$T/out")"
git fetch -q origin; git checkout -q -B g3 "$(rgit log -1 --format=%H --grep='^feat(test): g2$' main)"; echo 9 > m2; printf 'm1 10\nm2 9\nm3 10\n' > base.txt; echo 12 > m3; cm g3; m0=$(main)
run; [ "$rc" = 12 ] && [ "$(jf result)" = gate-diff ] && [ "$(main)" = "$m0" ] || fail "a worsening baseline was not stopped (exit $rc): $(cat "$T/out")"
# 6. a failing Tier 1 lands nothing
sync t; touch BAD; cm t; m0=$(main); run; [ "$rc" = 13 ] && [ "$(jf result)" = tier1 ] && grep -q "nothing lands" "$T/out" && [ "$(main)" = "$m0" ] || fail "a failing Tier 1 was not stopped (exit $rc): $(cat "$T/out")"
# 7. main moves during Tier 1: re-rebased, Tier 1 runs again, then it lands on top of the newcomer
sync mv1; echo mv1 > mv1.txt; cm mv1; rm -f "$T/t1n"; git -C "$T/other" pull -q origin main
printf 'git -C %s/other pull -q origin main && echo late > %s/other/late.txt && git -C %s/other add -A && git -C %s/other commit -qm "feat(test): late" && git -C %s/other push -q origin HEAD:main\n' "$T" "$T" "$T" "$T" "$T" > "$T/hook.1"
run; [ "$rc" = 0 ] && grep -q "moved during Tier 1" "$T/out" && [ "$(cat "$T/t1n")" = 2 ] && [ "$(showf late.txt)" = late ] && [ "$(showf mv1.txt)" = mv1 ] || fail "main moving during Tier 1 was not handled (exit $rc): $(cat "$T/out")"
rm -f "$T/hook.1"; git -C "$T/other" pull -q origin main
# 8. a refused push is retried, never forced; three refusals stop it with nothing landed
sync p1; echo p1 > p1.txt; cm p1; echo 1 > "$T/reject"; run; [ "$rc" = 0 ] && grep -q "was refused" "$T/out" && [ "$(showf p1.txt)" = p1 ] || fail "a refused push was not retried (exit $rc): $(cat "$T/out")"
sync p2; echo p2 > p2.txt; cm p2; m0=$(main); echo 3 > "$T/reject"; run; [ "$rc" = 1 ] && grep -q "refused three times" "$T/out" && [ "$(main)" = "$m0" ] || fail "three refused pushes did not stop it (exit $rc): $(cat "$T/out")"; echo 0 > "$T/reject"
# 9. the hold marker: stops it at the start; one set during Tier 1 stops it right before the push; released, it lands
sync h; echo h > h.txt; cm h; m0=$(main); "$land" --hold "owner says wait" >/dev/null
run; [ "$rc" = 14 ] && [ "$(jf result)" = held ] && grep -q "owner says wait" "$T/out" && [ "$(main)" = "$m0" ] || fail "a hold marker did not stop it (exit $rc): $(cat "$T/out")"
"$land" --release >/dev/null; rm -f "$T/t1n"; printf 'echo mid-flight > %s\n' "$common/hermes-prospecta-land-hold" > "$T/hook.1"
run; [ "$rc" = 14 ] && [ "$(main)" = "$m0" ] || fail "a hold set during Tier 1 did not stop the push (exit $rc): $(cat "$T/out")"
"$land" --release >/dev/null; rm -f "$T/hook.1"; run; [ "$rc" = 0 ] && [ "$(showf h.txt)" = h ] || fail "after release the commit did not land: $(cat "$T/out")"
# 10. main red (tier1 of its latest push failed) blocks a feature commit, lets a fix(ci): commit through
printf '5\tabc123abc123\tfeat: broke it\n' > "$T/latest"; printf 'tier1\tfailure\ntier2\t\n' > "$T/jobs"
sync rd; echo rd > rd.txt; cm rd; m0=$(main); run; [ "$rc" = 10 ] && [ "$(jf result)" = blocked-red ] && [ "$(main)" = "$m0" ] || fail "a red main did not block (exit $rc): $(cat "$T/out")"
sync fx; echo fx > fx.txt; cm fx "fix(ci): repair main"; run; [ "$rc" = 0 ] && [ "$(showf fx.txt)" = fx ] || fail "a fix(ci): commit did not land on a red main (exit $rc): $(cat "$T/out")"
printf 'tier1\tsuccess\n' > "$T/jobs"; sync ok; echo ok > ok.txt; cm ok; run; [ "$rc" = 0 ] || fail "a green main blocked: $(cat "$T/out")"
rm -f "$T/latest" "$T/jobs"
# 11. land-time checks: a bad subject stops; >150 production lines stop without a Diff-Budget trailer, land with one
sync k1; echo o > o.txt; cm k1 "update stuff"; m0=$(main); run; [ "$rc" = 1 ] && grep -q "Conventional" "$T/out" && [ "$(main)" = "$m0" ] || fail "a bad subject did not stop: $(cat "$T/out")"
sync k4; seq 1 160 > k4.py; cm k4; run; [ "$rc" = 1 ] && grep -q "160 production lines" "$T/out" && [ "$(main)" = "$m0" ] || fail "160 production lines without a waiver did not stop: $(cat "$T/out")"
sync k5; seq 1 160 > k5.py; git add -A; git commit -qm "feat(test): k5" -m "Diff-Budget: split next"; run; [ "$rc" = 0 ] && grep -q "waived: split next" "$T/out" || fail "a waived budget did not land (exit $rc): $(cat "$T/out")"
# 12. bad input: nothing to land (2), an unknown flag (2), a merge commit in the range (2)
sync n; run; [ "$rc" = 2 ] && grep -q "nothing to land" "$T/out" || fail "an empty range was not refused (exit $rc): $(cat "$T/out")"
run --nonsense; [ "$rc" = 2 ] || fail "an unknown flag was accepted"
sync mg; echo mg > mg.txt; cm mg; git checkout -q -b side HEAD~1; echo s > s.txt; cm s; git checkout -q mg; git merge -q --no-ff side -m "feat(test): merge" 2>/dev/null; run; [ "$rc" = 2 ] && grep -q "merge commits" "$T/out" || fail "a merge commit was not refused (exit $rc): $(cat "$T/out")"
[ -z "$(awk '$1 != "refs/heads/main"' "$T/updates")" ] || fail "a ref other than main was updated: $(cat "$T/updates")"
echo "tier1: land-direct: lands direct to main (no branch, no PR, no force), one coordinator: race, conflict, baselines, tier1 failure, main moving, refused push, hold, red main, commit checks: ok"
