# Collect & Review

Teleoperate the arms to record demonstration episodes, then sanity-check them before
training. Assumes the station is up — see [hardware.md](hardware.md); run everything from
the repo root with the venv active. Next step: [training.md](training.md).

Collect and Review need the `camera` and `gui` extras. Video encoding is a base dependency
(PyAV), so there is no system `ffmpeg` to install for recording or previews. Add
`--extra convert` if you'll also build LeRobot/ABC datasets here — see
[What to sync for what](../README.md#extras-table):

```bash
sudo apt install build-essential python3-dev   # i2rt compiles ruckig from source
uv sync --extra camera --extra gui
yam-abc-gui          # http://localhost:8042 ; the Collect tab opens by default
```

## Top bar (all tabs)

| Item | Meaning |
|---|---|
| **Collect / Train / Deploy / Review** | Switch tabs. |
| **ARM** | Arm/leader status once live (e.g. `ARM 2` = two follower arms, or the last error). `ARM —` before go-live. |
| **CAM** | Camera health, e.g. `CAM 3/3` (all streaming), turning red when any camera is not. Click for per-camera detail, the `/dev/video` holders, a kill button, and **Reset Cameras**. |
| **GPU** | **GPU 0**'s memory used / total (percent) · compute utilization, from `nvidia-smi`; reads `GPU n/a` when `nvidia-smi` reports nothing (`GPU —` is just the placeholder before the status feed connects). On a multi-GPU box it still tracks physical GPU 0 while a job may have been pinned to another card — the job's own `[yam-abc] pinned to GPU(s) …` log line names the one it got. Click it to list the processes holding the cards and kill a squatter — that is the recovery path when **Launch Train** or **Start Server** is refused for low free VRAM. |
| **connecting… / live** | Status-feed websocket state (green dot = connected). |

## Collect tab

A left **Station** rail (editable config) + a camera **stage** (previews) + the bottom
**tray** (live controls). The cameras open as soon as the page loads, so previews go live
immediately; the **robot, CAN buses and leaders** aren't touched until you press **Start
Teleop**, so edit the rail first.

Typical flow:

1. In the **Station** rail, set the **Task** name (e.g. `put the bottle in the bin`) — it
   becomes the episode's label and the training prompt, and is **required to record**.
2. Click **Start Teleop** — the followers ease to the leaders' pose, then track.
3. Drive with the teaching-handle buttons (or the on-screen ones):
   - **Top button** — sync on/off (follower mirrors leader).
   - **Second button** — start/stop recording an episode.
4. Perform the task, stop recording, repeat. The **Episodes** counter shows how many are
   saved for the current task.
5. **E-STOP** halts all motion; **Reset Session** → **Start Teleop** to re-arm.

Episodes are saved under `data/episodes/<task-slug>/<YYYYmmdd_HHMMSS>_<id>/` in
YAM-ABC-Reproduce's default format. The task name is **slugified** — lowercased, spaces and
punctuation to `_` — so `put the bottle in the bin` becomes `put_the_bottle_in_the_bin`, and
that slug is what the Train tab's task field and `yam-abc-convert data/episodes/<task>` want.

Headless alternative: `yam-abc-teleop --record "<task>" --seconds 15` records one episode — but
bring the CAN buses up yourself first (`bash third_party/i2rt/scripts/reset_all_can.sh`), since
unlike **Start Teleop** it does not reset CAN. Add `--mock` to exercise the path with no
hardware.

### Station rail (left)

Edits apply on the next Start Teleop. **Preview** additionally saves the camera rows back to
the roster file — whatever `cameras_config:` in the station YAML points at, `configs/cameras.yaml`
by default (or `--cameras`). Start Teleop applies camera edits for the session only, without
persisting them.

| Section | Field | What it is |
|---|---|---|
| **Robot** | type | Follower arm i2rt type (`yam_left`, `yam_right`, …). **+ Add robot** for a second arm (bimanual). |
| | gripper | Gripper type for that arm (e.g. `flexible_4310`). |
| **Controller** | type | Leader type driving a robot (`passive_gello_*` desk GELLO, `mobile_gello_*` mobile rig, `yam_lead_*` motorized). **+ Add controller** to pair another leader. |
| | controls robot | Which follower this leader drives. |
| **Cameras** | name / type / serial | One row per camera. **+ Add camera** to add; **↻ Detect** rescans connected cameras and fills serials; the trash icon removes a row. The **name doubles as the camera role** (there is no separate role field), and the role becomes the dataset key — so keep names to `top` / `left` / `right` / `wrist`. Other names produce off-schema keys the LeRobot/ABC converters and the policies won't recognize. |
| **Output** | format | On-disk recording format (`default`). Conversion to LeRobot/ABC happens later — see [training.md](training.md). |
| | save_root | Where episodes are written (default `data/episodes`). |
| **Task** | task name | The episode's label **and** the training prompt. **Required before recording.** |

### Bottom tray

| Group | Control | What it does |
|---|---|---|
| **Teleop** | **Start Teleop** | First click goes live: brings the CAN buses up — **follower *and* leader**, two per arm — builds the arms, and enables sync (followers ease to the leaders, then mirror them). Click again to stop sync. The physical **top** handle button toggles the same. (A rollout needs only the follower buses; see [deploy.md](deploy.md).) |
| **Record** | **Start Recording** | Start/stop recording one episode into `save_root/<task>/`. Needs a Task name. The lamp reads **idle** or **recording**; the physical **2nd** handle button toggles the same. |
| **Handle** | TOP · sync / 2ND · rec | Live indicators: light up when the handle's top (sync) or second (record) button is pressed. |
| | trigger bar | The teaching-handle trigger position (gripper open↔closed). |
| **Episodes** | count | Completed episodes recorded for the current task. |
| **Setup** | **Preview** | Open the cameras for framing **without** the robot/CAN/teleop. |
| | **Zero L / Zero R** | Zero the left/right passive-GELLO **leader** at its current pose (writes the encoder EEPROM). Hold the leader at the follower's home pose first. Sync is turned **off** before the write and stays off — press **Start Teleop** again to resume mirroring (no rebuild needed; the zero lives in the encoder EEPROM and takes effect immediately). See [hardware.md](hardware.md#4-leader-zero-calibration). |
| **Recovery** | **Reset CAN** | Bounce the CAN buses at the OS level — for a wedged/flaky bus. Does not touch the arms. |
| | **Power Off Arms** | De-energize both follower motors (torque off) — the arms go **limp**, so support them first. |
| | **Reset Session** | Recover after E-STOP or a failed start without restarting the GUI: stops the loop, releases the arms, and resets CAN so the next Start Teleop rebuilds cleanly. Also the **only** way back to teleop after a rollout, which releases the leaders (see [deploy.md](deploy.md)). Note it relaxes the arms to zero torque — they go **limp**. |
| **Session** | **End Session** | E-stop the arms and stop deploy jobs, releasing the hardware. The GUI itself keeps running — press **Start Teleop** again (after **Reset Session**) to re-arm; no `yam-abc-gui` restart needed. |
| — | **E-STOP** | Immediate stop: cease commanding and hold every arm in place (**still powered**), and stop any running deploy job. Any episode still recording is **discarded** (the partial folder is deleted), so stop recording first if you want to keep the take. Recover with **Reset Session** → **Start Teleop**. |

> **E-STOP vs Power Off Arms:** E-STOP keeps the motors energized so the arm *holds its
> pose*; Power Off Arms cuts torque so the arm *goes limp* (and will drop if unsupported).

## Review tab

Sanity-check recorded episodes before training — scan for missing frames, frozen video, or
implausible actions, and delete bad episodes so they don't reach training.

| Control | What it does |
|---|---|
| **Refresh** | Re-scan episodes on disk. |
| **source** filter | `all` / `dataset` (teleop recordings) / `rollout` (deploy recordings). |
| **task** filter | Limit the list to one task. |
| episode list | Click an episode to load it. The trash icon deletes it. |
| **Play** / scrub / frame counter | Play the synced camera videos, scrub to a frame, and see `frame / total`. |
| **arm** select | Which arm's joint/gripper curves to plot. |
| joint plot | The recorded **action** (leader command) per arm joint plus the gripper command, over the episode — axes read `action joints (rad)` / `gripper [0,1]`. Follower state is recorded but not plotted. |
| flags | Auto-computed quality flags for the episode. |

<a id="video-encoding"></a>
## Video encoding

Episodes are written as **H.264 MP4** (`<role>-images-rgb.mp4`), one file per camera, encoded
once when you press stop rather than per frame during teleop.

Which encoder does the work is **probed at startup, not configured**. `data/codec.py` tries
`h264_nvenc` first and falls back to `libx264`, and it decides by actually opening the codec
and pushing a frame through it. A name check is not enough: `h264_nvenc` is listed on any box
whose FFmpeg was *built* with NVENC, including ones with no NVIDIA card, where it only fails
once encoding starts. Decoding follows the same rule and uses NVDEC when it is genuinely
available. The first encode of a process logs what it settled on:

```
[yam-abc] video encoder: h264_nvenc
[yam-abc] video decode: cuda (hardware)
```

Everything runs through PyAV, which bundles its own FFmpeg — so none of this depends on the
system `ffmpeg`, which on Ubuntu 22.04 is 4.4.2 and cannot decode H.264 on Blackwell cards at
all. Two consequences worth knowing:

- **NVENC will not encode anything narrower than ~145 px.** Real camera frames are far above
  that, but if one is not, the writer says so and uses `libx264` for that stream:
  `[yam-abc] h264_nvenc refused 96x96; using libx264`.
- **Converted datasets can differ by machine.** `--to lerobot` hands LeRobot the probed
  encoder, so a GPU box writes H.264 and a CPU box writes LeRobot's default AV1. Both decode
  fine and the codec is recorded in the dataset's `meta/info.json`.

To pin it — for a reproducible dataset, or to rule the GPU out while debugging:

```bash
YAM_ABC_VIDEO_ENCODER=libx264 yam-abc-gui     # auto (default) | libx264 | h264_nvenc
YAM_ABC_VIDEO_HWDECODE=0 yam-abc-gui          # auto (default) | 0 | 1
```

Setting either to an explicit value makes it an assertion rather than a preference: asking for
hardware that is not there fails loudly instead of quietly running on the CPU.

Episodes recorded before this change are MPEG-4 Part 2 and stay readable — the Review tab
transcodes those to a cached `*.h264.mp4` for the browser, and skips that entirely for H.264.
