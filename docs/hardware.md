# Hardware setup

Bring a YAM bimanual station from bare machine to "teleop tracks smoothly." Install the
software first — venv, i2rt build, CAN sudoers — via the [README](../README.md#install);
this guide covers the physical wiring and calibration. For the end-to-end data → train →
deploy flow, continue with [training.md](training.md) and [deploy.md](deploy.md).

## 1. CAN buses

The YAM arms and passive-GELLO leaders talk over CAN. Bus names must match
`configs/station_yam.yaml`:

| Role | Bus name | Config type |
|---|---|---|
| Left follower | `can_left` | `yam_left` |
| Right follower | `can_right` | `yam_right` |
| Left leader | `can_lead_l` | `passive_gello_left` or `mobile_gello_left` |
| Right leader | `can_lead_r` | `passive_gello_right` or `mobile_gello_right` |

The leaders this station ships are **passive GELLO** arms — motorless CAN encoders that
broadcast on id `0x50F`. `mobile_gello_*` is the same hardware on a mobile rig, so it shares
the driver and buses and differs only in joint directions (see §2).
`yam_lead_left` / `yam_lead_right` name a *motorized* YAM leader
instead, and picking them builds the wrong driver: `yam-abc-doctor` then FAILs that bus with
"config expects motors but the bus carries a GELLO encoder broadcast".

The GUI brings the buses up for you on **Start Teleop**, and on the Deploy tab's **Load & Run**
(that's what the sudoers step enables). Teleop opens all four — follower *and* leader per arm;
a policy rollout opens only the two followers. To do it by hand:

```bash
sudo ip link set can_left   up type can bitrate 1000000
sudo ip link set can_lead_l up type can bitrate 1000000
```

If an arm is unresponsive, use **Reset CAN** in the GUI (or re-run the `ip link` command).

### Naming the adapters (udev)

> Fastest path: `python scripts/setup_can_udev.py` walks you through the
> unplug/replug identification and writes `/etc/udev/rules.d/90-can.rules`
> for you. The steps below are the manual equivalent.

Those stable names (`can_left`, `can_lead_l`, …) come from **udev rules** that map each
USB-CAN adapter's serial to a fixed interface name — otherwise Linux assigns `can0/can1/…`
in plug order, which changes between boots. Set this up once (see i2rt's CAN setup guide
for the rule template). To find which physical adapter is which:

1. Plug in **one** adapter and run `ip -br link | grep can` to see the new interface, then
   read its serial with `udevadm info -a -p /sys/class/net/<iface> | grep -i serial`
   (`ip -br link` prints name/state/MAC only — never the USB serial). Label that cable
   (e.g. "left follower").
2. Repeat one adapter at a time, then write a `/etc/udev/rules.d/*.rules` line per serial
   assigning the name from the table above.

After the rules are in place, `ls /sys/class/net | grep can` should show the named
interfaces, and the bus names in `configs/station_yam.yaml` will resolve correctly.

## 2. Robots & controllers

Edit `configs/station_yam.yaml`:

- Each `robots` entry = a follower arm + its gripper.
- Each `controllers` entry = a leader and the robot it `controls`.

The GUI **Station** rail can override robots/controllers for the current session, but only
the camera roster is written to disk (Preview/Detect save `configs/cameras.yaml`).
Robot/controller edits made in the rail are lost on the next go-live or GUI restart, because
the station YAML is re-read from disk each time — so anything you want to keep belongs in the
file.

A passive-GELLO leader is **not** an identity map onto the follower: it goes through the
per-joint `leader_joint_signs` and the `leader_gripper` range in `configs/station_yam.yaml`.
(Only a motorized `yam_lead_*` leader maps straight through.) CAN channels are derived from
the type (`yam_left`→`can_left`, `passive_gello_left`→`can_lead_l`).

A desk GELLO's two arms share `leader_joint_signs`. A **mobile GELLO**'s right arm is
mirrored, so `mobile_gello_left` / `mobile_gello_right` each carry their own signs in code —
`[-1, 1, 1, 1, -1, -1, -1]` and `[-1, -1, -1, -1, -1, -1, 1]` (arm joints then gripper) —
which override the station-wide list, since one list can't describe two mirrored arms. Both
tables are copied verbatim from the fleet's own `passive_gello.py` and must track it. To
retune one leader, add `joint_signs: [...]` to its `controllers` entry; that wins over both.

### Leader trigger

The trigger is normalized by **distance from zero**, against `min(|closed_rad|, 0.67)`:
released (`0` rad) is `1.0` open, `|angle|` at the range is `0.0` closed. Two consequences:

- **Trigger direction doesn't matter.** A desk trigger counts negative when squeezed and a
  mobile one counts positive; both map to the same curve. The gripper element of each
  `joint_signs` table is therefore inert — which is why the two mobile sides can disagree
  on it (`-1` vs `+1`) with no effect.
- `open_rad` in `leader_gripper` is informational only; the released position is always `0`.

A signed mapping would clip whichever direction fell outside the range to a constant, with
nothing logged. Watch the **RAW** / **CAL** rows in the GUI's per-arm panel to confirm a
trigger is actually sweeping.

### Follower gripper travel

**Every** YAM gripper type (`flexible_4310`, `linear_4310`, `linear_3507`, `crank_4310`)
ships `gripper_limits: null` + `needs_calibration: true`, so i2rt re-derives each follower
gripper's `[closed, open]` motor range **on every build** — it pushes the motor into both
stops with a test torque and takes the stall positions. The normalized `[0, 1]` trigger is
then mapped onto exactly that span (`motor_pos = norm * (open - closed) + closed`).

Because it is measured per arm, per run, from wherever the jaws happen to be sitting, two
arms can disagree. If one gripper's jaws start blocked (holding something, resting closed)
or the test torque can't overcome its friction, stall detection fires almost immediately,
the span collapses, and that arm's gripper barely responds to the trigger while the other
tracks fine. Nothing errors — the mapping just compresses.

Each arm logs what it settled on at startup, and warns when the span is implausibly small
against the ~6.57 rad motor stroke:

```
left follower gripper (flexible_4310): limits=[...] (auto-detected this run)
right follower gripper (flexible_4310): limits=[...] (auto-detected this run)
WARNING left follower gripper travel is only 0.083 rad ...
```

If the two spans differ substantially, the narrow one mis-calibrated. Clear the jaws, go
live again to get a clean measurement, then freeze the good values per arm:

```yaml
robots:
  - {type: yam_left,  gripper: flexible_4310, gripper_limits: [<closed>, <open>]}
  - {type: yam_right, gripper: flexible_4310, gripper_limits: [<closed>, <open>]}
```

A pinned range skips auto-calibration entirely, so the mapping is identical on every run.
Re-measure after changing or reseating a gripper — a range measured on one gripper does not
transfer to another. The GUI rail can't edit these; it carries the pin forward only while
that arm's gripper type is unchanged.

## 3. Cameras

List what's connected (each line shows a serial):

```bash
yam-abc-cameras
```

Discovery lives in the `camera` extra. Without it (`uv sync --extra camera …`) this prints
"(none found)" even with cameras plugged in — indistinguishable from a cabling fault.

Put a distinct `serial:` in `configs/cameras.yaml` for each camera (required when
several are the same model, e.g. multiple RealSense). Supported `type`s: `realsense`
(RealSense), `decxin_mono` (single-lens UVC cam), `decxin_stereo` (side-by-side stereo UVC
cam), plus `mock` for no-hardware dev. You can also add/preview cameras from the GUI Station
rail.

## 4. Leader zero calibration

Do this the first time, whenever a leader was moved/re-seated, or if an arm tracks the
wrong way.

**Set the home zero** — hold the leader in the pose that should map to the follower's
home, **with the trigger fully released**, then:

- GUI: **Zero L** / **Zero R**, confirm; **or**
- Terminal:
  ```bash
  python scripts/calibrate_gello.py zero --controller left
  python scripts/calibrate_gello.py zero --controller right
  ```

This zeroes every joint encoder including the gripper's. The trigger has to be released
because its normalization measures distance from zero (above), so zero *is* released. A
gripper encoder carrying some other zero reads a constant offset, and once that offset
exceeds the travel the normalized value sticks at one end permanently — a gripper stuck at
`0.00` or `1.00` while the other side is fine is almost always this. Confirm with the
**RAW** row in the per-arm panel: a released trigger should read ≈ `0.000`.

**Check joint direction (optional)** — live readout while you wiggle each joint:

```bash
python scripts/calibrate_gello.py monitor --controller left
```

The readout uses whichever leader variant that side is configured as, and prints its type
and resolved signs.

If a follower joint moves the *opposite* way during teleop, flip that joint's sign in
`configs/station_yam.yaml` (`leader_joint_signs`, or that controller's `joint_signs`), then
press **Reset Session** and **Start Teleop** again (or restart `yam-abc-gui`). Stop/Start
Teleop on its own only toggles sync — the signs are baked into the leader when it is built,
so the old ones survive.

## 5. Verify teleop

Check the whole station first — read-only, and safe to run even while teleop is live:

```bash
yam-abc-doctor                    # add --no-listen to skip the 1s-per-bus passive CAN listen
```

It flags unnamed `can0/can1` adapters, buses the config expects but that are missing, cables
swapped between a leader and a follower (from each bus's traffic fingerprint), RealSense
serials in `cameras.yaml` that aren't on USB, a missing passwordless-CAN sudoers rule, and
missing `ffmpeg` — each with the fix command. Exit code is 1 if any row FAILs. Two `sys.backend.*`
rows always WARN: only one policy backend fits the venv at a time, so the two you haven't
synced are reported missing by design.

```bash
yam-abc-gui                       # http://localhost:8042, Collect tab, previews go live
```

With the buses up, confirm the **YAM leader floats freely in your hand**. Press **Start Teleop**: the
follower eases to the leader's pose, then tracks. It should **never snap**. Watch the live
previews and the ARM badge, plus the handle's TOP/2ND lamps and the trigger bar in the tray.

Prefer headless? `yam-abc-teleop --seconds 15` runs teleop from the terminal — but bring the
buses up yourself first (the `ip link` commands above, or
`bash third_party/i2rt/scripts/reset_all_can.sh`). Unlike **Start Teleop**, it does not reset
CAN for you, so on a freshly booted station it fails while opening a down interface.

## Safety

Keep a hardware e-stop within reach. The GUI **E-STOP** halts the control loop and any
deploy process immediately; `Ctrl-C` does the same for terminal runs. It also **discards any
episode still recording**, so stop recording first if you want to keep the take. To re-arm, the
GUI prompts **Reset Session** → **Start Teleop**, and that is the path to prefer: Reset Session
releases the arms and resets CAN before the rebuild. (**Start Teleop** alone also rebuilds —
E-STOP clears the live flag, so it goes through a full `go_live` rather than just re-enabling
sync — but it skips that cleanup.)

After a **policy rollout** the leaders have been released, so **Start Teleop alone won't do**:
it refuses with a message pointing at Reset Session, which is what rebuilds them. Reset Session
relaxes the arms to zero torque first — support them before clicking it.
