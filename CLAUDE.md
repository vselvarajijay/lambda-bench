# Agent instructions

Every code change follows this workflow. A task is not done until its PR is open.

## 1. Plan and branch

- For multi-step work, create or continue a plan in `.plans/`. See `.plans/README.md`.
- Do the work on a new branch. Never commit directly to `main`.

## 2. Write tests

- Add or update tests for every behavior you add or change.
- For a bug fix, add a test that fails without the fix and passes with it.
- Cover failure paths and edge cases, not only the happy path.

## 3. Verify it works

- Run the full test suite, plus any lint, type-check, and build steps the project has.
- Exercise the change for real where possible, for example by running the code or invoking the command, rather than relying on tests alone.
- Do not skip, disable, or weaken a test to get a passing run. Fix the cause, or report the failure.

## 4. Submit a PR

- Commit the work, push the branch, and open a PR against `main` with `gh pr create`.
- In the PR description, state what changed, why, and how it was verified, including the commands you ran and their results.
- List anything you could not verify or that still fails.
- Do not merge the PR unless asked to. The `ci-watcher` agent handles merging under its own rules.
