# Contributing

Ground rules distilled from field debugging on two real stations. Every rule
below exists because violating it cost hours on hardware — please keep them.

## Design rules (review checklist)

1. **Periodic background checks must never touch a hardware bus.**
   A 5-second camera-health poll that enumerated RealSense devices over USB
   caused a metronomic teleop stutter (the CAN adapters share USB topology).
   Pollers read cached in-memory state; bus-touching probes run only on
   explicit user action (opening a panel, pressing a button).

2. **Every failure path must tear down what it built.**
   A half-built arm unit that keeps its control thread alive hammers the bus
   at kHz rates and makes every retry fail differently (see: "loss
   communication" roulette). Build → fail → roll back, always.

3. **Processes release hardware on the way out.**
   SIGTERM/shutdown hooks must stop control loops and close camera pipelines;
   a hard-killed GUI used to leave RealSense wedged until a hardware reset.
   Prefer `kill -TERM` in scripts and docs; never rely on `kill -9` cleanup.

4. **Stopping has three distinct meanings — don't blur them.**
   - **E-STOP**: stop commanding, motors HOLD pose (a held payload must not drop).
   - **Reset Session / relax**: zero torque, arm goes limp, session rebuildable.
   - **Power Off Arms**: physically de-energize the followers.
   Any new stop-like feature must state which semantic it implements.

5. **Fail with the user's next action in the message.**
   "fail to communicate with motor 1" cost hours; the actual cause (motor
   undervoltage, error code 0x9 in the reply frame) was on the bus all along.
   Decode error codes, name the config file to edit, list the valid options.

6. **Give every long-running phase a visible state.**
   JIT compiles, norm-stats passes, model downloads and server startups all
   look like hangs without a badge, a log line with an expected duration, or
   a progress hint. If it can take >10s, it must say so in the UI.

7. **Guard the GPU before starting work on it.**
   Check free VRAM before launching a training/serving job (a colleague's
   13 GB squatter turns a clean run into a one-minute OOM), and refuse to
   double-start a server onto a busy port.

8. **Safety clamps are speed-based.**
   Action limits are specified in rad/s (control-rate invariant), applied
   per-step, truncating with a rate-limited warning. Never pass raw policy
   output to the arms unclamped.

## Practicalities

- Station-local files (`configs/station_yam.yaml` hardware types, camera
  serials, udev rules) never go in commits — they differ per site.
- The `freemani` branch names in `.gitmodules` and the legacy `FREEMANI_*`
  env aliases in `gui/builders.py` stay until the policy forks migrate.
- Passive CAN diagnostics live in `scripts/` (`check_can_usb_quality.py`,
  `monitor_can_reply_latency.py`); they only listen and are safe during
  teleop. Use them before blaming software for bus dropouts.
