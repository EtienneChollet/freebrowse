# Agent Workflow

This defines the standard workflow for all coding agent sessions in this project.

---

## 1. Linear Issue

Before starting any work:

1. **Ask** if there is a Linear issue associated with the task.
2. If yes, use `/linear-context` or the Linear MCP to pull full context (description, status, related issues).
3. If no, ask if we should create one before proceeding. Use `/linear-issue` or the Linear MCP to create it.

Every non-trivial change should be traceable to a Linear issue.

---

## 2. Branch and Worktree

1. Copy the **git branch name** from the Linear issue (Linear auto-generates these, e.g. `epc28/epc-1742-fix-compile-errors`).
2. Create a git worktree in the `worktrees/` directory at the repo root:
   ```bash
   git worktree add worktrees/<branch-name> -b <branch-name>
   ```
   If the branch already exists on the remote:
   ```bash
   git fetch origin <branch-name>
   git worktree add worktrees/<branch-name> <branch-name>
   ```
3. All work happens inside the worktree. The main checkout stays clean.

---

## 3. Do the Work

Implement changes in the worktree following all project and global CLAUDE.md standards.

### Commits

**Format:** Every commit message MUST be prefixed with a tag in square brackets:

| Tag | Use when |
|-----|----------|
| `[fix]` | Bug fixes |
| `[feat]` | New features or functionality |
| `[refactor]` | Code restructuring with no behavior change |
| `[docs]` | Documentation changes |
| `[tests]` | Adding or updating tests |

Examples:
```
[fix] resolve race condition in volume loading
[feat] add model selection dropdown to toolbar
[refactor] extract inference pipeline from server.py
[tests] add coverage for QA rating endpoint
[docs] update API route descriptions
```

Commit messages should be one line, focused on the "why".

### Attribution Ban

**NEVER** include any of the following in commit messages, code comments, PR descriptions, Linear comments, or anywhere else:

- "Claude Code", "Claude", "Anthropic", "Codex", "OpenAI", "AI-generated", "AI-assisted"
- `Co-Authored-By` lines referencing any AI tool
- Any other attribution to an AI coding assistant

This applies to ALL outputs: commits, comments, docstrings, PR bodies, Linear updates. No exceptions.

---

## 4. Linear Updates

Post progress updates to the Linear issue as work proceeds (use `/linear-update`). Keep updates focused on what changed and why, relevant to the issue only.

---

## 5. Wrap Up

When all work is complete:

1. Ensure all commits are clean and atomic.
2. **Ask the user** if we should merge into main.
3. If yes, merge locally:
   ```bash
   # From the worktree, make sure everything is committed
   # Then switch to main in the main checkout and merge
   cd /autofs/cluster/dalcalab2/users/epc28/projects/freebrowse_bhim
   git checkout main
   git merge <branch-name>
   git push origin main
   ```
   Then clean up the worktree:
   ```bash
   git worktree remove worktrees/<branch-name>
   git branch -d <branch-name>
   ```
4. If not yet, leave the worktree in place for future sessions.
