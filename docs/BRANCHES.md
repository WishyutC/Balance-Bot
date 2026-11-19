# Segment branches

Each segment branch contains the corresponding original directory at the repository root, plus shared documentation, verification tooling, and Git exclusions. The combined archive retains the original four-directory layout. Git history retains the archive commit as the split branches' parent.

| Branch | Original directory | Main purpose |
| --- | --- | --- |
| `segment/b-bots` | `b_bots` | Baseline voice assistant |
| `segment/b-bots-new` | `b_bots(new)` | Voice and typing selection |
| `segment/b-bots-new-pkongkha` | `b_bots_new_pkongkha` | Mood recognition; includes its historical `old/` snapshot |
| `segment/b-bot-frontend` | `b_bot_frontend` | React dashboard and accompanying Python code |

Clone one branch into a separate directory:

```bash
git clone --single-branch --branch segment/b-bots https://github.com/WishyutC/Balance-Bot.git balance-bot-baseline
```

Separate clones avoid mixing local environments, models, and generated files between variants. Provision models and dependencies separately in each clone; they are not part of the branches. See `TEST_REPORT.md` for verification results and known failures.
