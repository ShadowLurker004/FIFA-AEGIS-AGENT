# FIFA-AEGIS: Adaptive Expected-Goal Intelligence System

![Competition](https://img.shields.io/badge/Conv__Cup-'26-blue?style=for-the-badge)
![Environment](https://img.shields.io/badge/Arena-2D%20Football%201v1-green?style=for-the-badge)
![Language](https://img.shields.io/badge/Python-Standard%20Library%20Only-yellow?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Development-orange?style=for-the-badge)

> **A tuned, rule-based 1v1 football agent for the Conv_Cup '26 AI Soccer Arena (MLFootball environment).**

FIFA-AEGIS is a deterministic tactical controller built for the official Conv_Cup '26 simulator. It separates **ball prediction, tactical decision-making, and safe action execution**, and it tunes the controller's parameters by running thousands of simulated matches on both sides of the pitch.

![FIFA-AEGIS Architecture](assets./Architecture.png)

> [!NOTE]
> This README describes the design, the workflow, and the real environment rules. The only performance numbers included are measurements of the **unmodified organizer starter bot** (Section 8). No result is claimed for FIFA-AEGIS until it has been measured the same way.

---

## 1. Overview and Design Thesis

The competition runs two independent bots on a small continuous pitch. Each bot gets one JSON observation per iteration and must answer within **2 seconds** using **only the Python standard library**.

Under those limits, FIFA-AEGIS follows four principles:

- **Goals decide matches.** The objective is net goal differential, not possession or distance covered.
- **Hand-written tactics first.** Interception, shooting, defending and obstacle avoidance are explicit geometry, so behavior is predictable and easy to debug.
- **Parameters are tuned, not guessed.** Constants such as kick power, shooting range and pressing offset are chosen by automated search over many seeds.
- **A valid action every turn.** Every decision passes a safety projector, and a deterministic fallback covers any failure.

```text
Net objective:   ΔG = Goals Scored − Goals Conceded
```

---

## 2. Environment Facts

All values come from the official `config/game.json`.

| Parameter | Value |
| --- | --- |
| Field | 100 wide × 140 tall (origin at bottom-left) |
| Goal width | 36, centered on each short end (x from 32 to 68) |
| Player radius / speed | 3 / 4 units per iteration |
| Ball radius / speed | 1.5 / 8 units per iteration |
| Possession radius | 5 |
| Kick distances (power 1 / 2 / 3) | 32 / 64 / 96 |
| Obstacles | 6 axis-aligned rectangles, 11 × 8, mirrored top and bottom |
| Match length | 400 iterations, or 7 total goals |
| Possession limit | 10 iterations, then the ball is released forward |
| Loose-ball restart | Drop-ball at midfield after 20 iterations without progress |
| Opening possession | `player_1` |

### Rules That Shape the Strategy

- **Sides:** Player 1 defends the bottom goal and attacks upward. Player 2 defends the top goal and attacks downward. The observation includes `attack_direction`.
- **Movement:** eight directions plus `STAY`. Diagonal and straight moves have the same speed.
- **Kicking:** only with possession. The ball then flies independently for the power's distance and bounces off walls and obstacles.
- **Interception:** a moving ball is picked up by any player within ball radius + player radius (4.5).
- **Stationary ball:** the nearest player within the possession radius (5) claims it.
- **Tackles:** a newly won ball is protected for its first 3 iterations. After that, a moving challenger in contact (within 6.15) takes the ball, unless the owner kicks that same iteration.
- **Possession timeout:** after 10 iterations of holding the ball, the engine kicks it forward at power 1.
- **After a goal:** both players return to their start positions and the conceding side gets possession.
- **No goalkeeper:** the open goal mouth is the only defense, so positioning matters.

---

## 3. Interface Contract

The bot is a process that reads one JSON observation per line from `stdin` and writes exactly one JSON action per line to `stdout`.

```json
{"move": "UP_RIGHT"}
```

```json
{"move": "UP", "kick": {"direction": "UP_LEFT", "power": 3}}
```

- Legal moves: `STAY`, `UP`, `UP_RIGHT`, `RIGHT`, `DOWN_RIGHT`, `DOWN`, `DOWN_LEFT`, `LEFT`, `UP_LEFT`.
- Kick power is an integer from 1 to 3. A kick without possession is ignored.
- Invalid, late, missing or malformed output becomes `STAY` and is recorded as an action error.

```diff
+ stdout  → one JSON action per observation
- stderr  → diagnostics only
```

> [!WARNING]
> Never print anything except the action JSON to `stdout`.

---

## 4. Competition Constraints

These come from `config/submission_policy.json` and the participant guide.

| Area | Rule |
| --- | --- |
| Dependencies | `allowed_dependencies` is empty, so standard library only (no NumPy, PyTorch or TensorFlow) |
| Forbidden imports | `os`, `pickle`, `subprocess`, `socket`, `urllib`, `http`, `ctypes`, `importlib`, `multiprocessing`, `shutil` and others |
| Forbidden calls | `eval`, `exec`, `compile`, `__import__`, filesystem-mutating `Path` methods |
| Model files | JSON and other plain data only. `.pt`, `.pth`, `.pkl`, `.pickle`, `.joblib` are rejected |
| Latency | One response within 2 seconds, at most 8,192 bytes |
| Archive | At most 100 MB, 2,000 files, 512 KB per source file |
| Required in ZIP root | `submission.json`, `README.md`, `requirements.txt` |
| Organizer assets | Do not ship the organizer RL model unless the published rules allow it |

**Tournament format:** double elimination. A tie has one to six games, alternates sides, and is decided on aggregate goals. An aggregate draw goes to a recorded seeded penalty shootout. Every game uses a hidden seed.

---

## 5. Architecture

Each decision runs the same short pipeline.

```text
OBSERVE → NORMALIZE → PREDICT → CHOOSE MODE → PLAN → PROJECT → JSON ACTION
```

![AEGIS Decision Loop](assets./Loop.png)

| Layer | Responsibility |
| --- | --- |
| **State normalizer** | Mirrors coordinates by `attack_direction`, so one policy plays both sides |
| **Ball predictor** | Projects the ball from its position, velocity and remaining kick distance, including wall and obstacle bounces |
| **Tactical manager** | Chooses a mode from the possession state (below) |
| **Action planner** | Picks movement, and a kick direction and power when in possession |
| **Safety projector** | Rejects moves that leave the field or hit an obstacle, and validates the JSON |
| **Fallback policy** | Simple deterministic behavior used if planning fails or errors |

### Tactical Modes

The game has three possession states (ours, theirs, loose), which map to four modes.

| Mode | When | Behavior |
| --- | --- | --- |
| `ATTACK` | We hold the ball | Carry forward, evade the defender, shoot when the lane is open |
| `DEFEND` | Opponent holds the ball | Press from the goal side and cut the shooting line |
| `INTERCEPT` | Ball is moving | Move to the predicted interception point |
| `CLAIM` | Ball is stationary | Take the shortest obstacle-free path to the ball |

### Planning Details

- **Shooting lane:** a shot is taken only when the straight path to the goal mouth does not cross an obstacle rectangle. Wall bounces are not relied on.
- **Kick power:** chosen from the distance to goal, and treated as a tunable parameter, since 32 / 64 / 96 units give the ball 4 / 8 / 12 iterations of flight.
- **Possession timeout:** the agent releases the ball on its own terms before the engine's forced kick at 10 iterations.
- **Obstacle avoidance:** candidate moves are checked against the obstacle rectangles and walls, and the move closest to the desired direction is chosen.
- **Fallback:** if any step raises an error, the bot returns a safe move toward the ball (or `STAY`), so the action is always valid.

---

## 6. Parameter Tuning and Training

The tactical controller exposes its important constants as parameters, for example kick power by distance, shooting range, pressing offset, dribble side-step, and defender-ahead thresholds.

### Tuning loop

1. Define a pool of opponents: organizer RL, the simple baseline, aggressive, counter-attack, and frozen earlier versions of our own bot.
2. Generate candidate parameter sets with random search.
3. Evaluate each candidate on hundreds of seeds, **on both sides**, with the official `config/game.json`.
4. Keep a candidate only if it also wins on a separate set of unseen seeds.

The simulator is dependency-free and fast. In a measured run, 300 matches took about 24 seconds, so a candidate can be scored in seconds.

### Optional learned layer

The kit includes `train_bot.py`, a pure-Python tabular reinforcement learner that writes a sparse JSON model. The starter policy uses it as an advisor and falls back to the tactical rules in unfamiliar states.

- Training episodes are shortened by the trainer (240 iterations, 5 goals). Final evaluation always uses the official 400 iterations and 7 goals.
- The learned model is promoted only if it beats the tuned rules on unseen seeds.

### Staged training schedule

![Training Curriculum](assets./Curriculum.png)

| Stage | Scenario | Goal |
| --- | --- | --- |
| **C0** | Navigation without obstacles | Reach targets without wall or obstacle traps |
| **C1** | Ball approach and straight shots | Reliable open-goal conversion |
| **C2** | Simple baseline opponent | Beat the readable baseline on both sides |
| **C3** | Organizer RL opponent | Positive goal differential |
| **C4** | Shared-policy self-play | Remove side bias |
| **C5** | Frozen snapshots of our own bot | Avoid regressions and overfitting |

---

## 7. Evaluation Protocol

Every version is judged on matches, not on one attractive game.

| Metric | Definition |
| --- | --- |
| **Win / Draw / Loss** | Counted per match, with both sides played on every seed |
| **Goal differential per match** | (Goals scored − goals conceded) / matches |
| **Action errors** | Must be zero in `validate_submission.py` |
| **Decision time** | Must stay far below 2 seconds |
| **Seed spread** | Variation of goal differential across unseen obstacle layouts |

Rules of thumb:

- Use separate training and evaluation seed sets.
- Always measure both Player 1 and Player 2, because Player 1 opens with possession.
- Keep the previous model as a frozen benchmark and compare on identical seeds.
- Change one major idea at a time.

---

## 8. Baseline Measurements (Organizer Starter Bot)

These numbers describe the **unmodified starter heuristics** from `submission_kit`, not FIFA-AEGIS. They give the bar to beat.

**Method:** official `config/game.json`, in-process engine, 150 seeds with both sides played (300 matches per row), matches of 400 iterations or 7 goals.

| Starter bot vs | W / D / L | Goals for / against | Goal diff per match |
| --- | --- | --- | --- |
| Organizer RL bot | 160 / 63 / 77 | 548 / 379 | +0.56 |
| Aggressive bot | 142 / 50 / 108 | 521 / 402 | +0.40 |
| Counter-attack bot | 158 / 79 / 63 | 716 / 261 | +1.52 |
| Simple baseline (mirror) | 113 / 74 / 113 | 521 / 521 | 0.00 |

**Kick-power sensitivity** (starter bot vs the organizer RL bot, same 300 matches):

| Starter kick power | W / D / L | Goal diff per match |
| --- | --- | --- |
| 3 (default) | 160 / 63 / 77 | +0.56 |
| 2 | 170 / 68 / 62 | +0.73 |
| 1 | 179 / 52 / 69 | +1.07 |

Takeaways: the organizer RL bot is beatable with simple rules, one constant already moves the result, and the mirror match is even, so other teams starting from the same kit will be close to this baseline.

> [!NOTE]
> Results use the organizer model bundled with the participant kit. Tournament seeds are hidden, so treat these as a guide, not a guarantee.

---

## 9. Repository Structure

```text
FIFA-AEGIS-AGENT/
├── assets/
│   ├── Architecture.png
│   ├── Loop.png
│   └── Curriculum.png
├── README.md                      (this file)
├── config/ soccer_env/ viewer/    (organizer kit, unchanged)
├── organizer_rl_bot/ reference_bot/
├── run_match.py live_viewer.py train_bot.py
├── validate_submission.py package_submission.py check_submission.py
├── tests/
├── tools/                         (planned: benchmark and tuning scripts)
└── my_team/                       (our submission; becomes the ZIP root)
    ├── submission.json
    ├── README.md
    ├── requirements.txt
    └── team_bot/
        ├── __init__.py
        ├── bot.py
        ├── policy.py
        └── models/
```

| Path | Purpose |
| --- | --- |
| `my_team/` | The competition submission. Its contents become the ZIP root |
| `my_team/submission.json` | Public team name and launch command |
| `my_team/team_bot/bot.py` | Protocol loop (stdin and stdout JSON lines) |
| `my_team/team_bot/policy.py` | FIFA-AEGIS decision logic |
| `my_team/team_bot/models/` | Optional learned model (JSON) |
| `my_team/README.md`, `requirements.txt` | Required by the ZIP checker |
| `tools/` | Planned benchmark and parameter-search scripts |

---

## 10. Quickstart

Run everything from the organizer kit folder (PowerShell).

### 10.1 Verify the kit

```powershell
python -m unittest discover -s tests -v
```

### 10.2 Play a match against the organizer bot

```powershell
python run_match.py --submission my_team/submission.json --opponent organizer-rl --seed 101
```

Use `--opponent simple` for the readable baseline.

### 10.3 Watch it in the browser

```powershell
python live_viewer.py --submission my_team/submission.json --seed 101
```

### 10.4 Train the optional learned layer

```powershell
python train_bot.py `
  --episodes 5000 `
  --opponents curriculum `
  --self-play-ratio 0.35 `
  --output my_team/team_bot/models/trained_policy.json
```

### 10.5 Validate the real process

```powershell
python validate_submission.py `
  --submission my_team/submission.json `
  --matches-per-side 2
```

The result must report zero participant action errors.

### 10.6 Package and check

```powershell
python package_submission.py my_team dist/my-team.zip
python check_submission.py dist/my-team.zip --report dist/my-team-report.json
```

Submit the exact ZIP that passed and record its SHA-256.

---

## 11. Submission Checklist

- Official tests pass.
- Final tuning and evaluation used the unmodified `config/game.json`.
- Both sides and many seeds were covered, with unseen validation seeds for selection.
- `stdout` contains only action JSON.
- Every action returns well inside the 2-second limit.
- Protocol validation reports zero action errors.
- The bot runs offline with the standard library only.
- `requirements.txt` lists no unapproved packages.
- The ZIP was made with the clean packager and passes the static checker.
- The final ZIP is unchanged after its hash is recorded.

---

## 12. Project Status

- [x] Official environment studied and rules documented
- [x] Starter bot benchmarked on 300 matches per opponent
- [ ] Benchmark and tuning scripts (`tools/`)
- [ ] Shooting-lane check against obstacle rectangles
- [ ] Kick power chosen from distance to goal
- [ ] Ball interception using predicted bounces
- [ ] Goal-side defensive positioning
- [ ] Possession-timeout management
- [ ] Automated parameter search with unseen-seed validation
- [ ] Optional learned layer (only if it beats the tuned rules)
- [ ] Final validation, packaging and static check

---

## 13. References

1. Kurach et al. (2020). **Google Research Football: A Novel Reinforcement Learning Environment.** AAAI 2020, 34(04), 4501–4510.
2. Lin et al. (2023). **TiZero: Mastering Multi-Agent Football with Curriculum Learning and Self-Play.** arXiv:2302.07515.
3. Song et al. (2023). **An Empirical Study on Google Research Football Multi-agent Scenarios.** arXiv:2305.09458.
4. **Conv_Cup '26 AI Soccer Arena participant kit and guides** (`participants/README.md`, `README_TRAINING_AND_SUBMISSION.md`), IIT (ISM) Dhanbad.

The Google Research Football and TiZero papers inspired the curriculum and self-play ideas. No code from those projects is used.

---

<p align="center">
  <b>FIFA-AEGIS — Adaptive Expected-Goal Intelligence System</b><br>
  <i>Built for the Conv_Cup '26 AI Soccer Arena.</i>
</p>
