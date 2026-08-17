"use strict";

const $ = (id) => document.getElementById(id);
const post = (url, body) =>
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  }).then((r) => r.json());

// --- tabs -------------------------------------------------------------------
document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.remove("on"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("on"));
    btn.classList.add("on");
    document.querySelector(`.panel[data-panel="${btn.dataset.tab}"]`).classList.add("on");
    // The control tray + arm telemetry belong to the Collect tab only.
    const collect = btn.dataset.tab === "collect";
    $("tray").style.display = collect ? "" : "none";
    $("armmon").style.display = collect ? "" : "none";
  };
});

// --- station config rail (editable form; applied to the session on Start Teleop) --
let OPTS = { controller_type: [], robot_type: [], gripper: [], camera_type: [], camera_role: [], data_format: [] };
// Live-hardware scan (/api/cameras/detect) so the rail's serial pickers offer real
// devices instead of hand-typed serials. Refreshed at boot and via the ↻ button.
let DEVICES = [];

// <option> tags from a list of strings or {value,label} objects.
function optionTags(values, selected) {
  return values
    .map((v) => {
      const val = typeof v === "string" ? v : v.value;
      const label = typeof v === "string" ? v : v.label;
      return `<option value="${val}"${val === selected ? " selected" : ""}>${label}</option>`;
    })
    .join("");
}
// CAN bus picker: choose a detected interface from the shared datalist or type another.
// Blank means "derive from type"; the placeholder shows the bus that would be opened.
function channelField(entry) {
  const dflt = entry.channel_default || "";
  return `<input class="inp" data-k="channel" list="can-list" value="${entry.channel || ""}"
    placeholder="${dflt || "(auto)"}"
    title="CAN bus for this device. Blank = derive from type${dflt ? " (" + dflt + ")" : ""}." />`;
}
function robotRow(rb) {
  return `<div class="frow" data-robot>
    <select class="sel" data-k="type">${optionTags(OPTS.robot_type, rb.type)}</select>
    <select class="sel" data-k="gripper">${optionTags(OPTS.gripper, rb.gripper)}</select>
    ${channelField(rb)}
    <button class="rm" data-rm title="remove row">×</button></div>`;
}
function controllerRow(ctrl) {
  return `<div class="frow" data-controller>
    <select class="sel" data-k="type">${optionTags(OPTS.controller_type, ctrl.type)}</select>
    <select class="sel" data-k="controls">${optionTags(OPTS.robot_type, ctrl.controls)}</select>
    ${channelField(ctrl)}
    <button class="rm" data-rm title="remove row">×</button></div>`;
}
// <option> tags for the serial picker: the detected devices, plus the currently
// configured serial if it isn't attached right now (so it isn't silently dropped),
// plus a blank "(auto)" that falls back to device-index discovery.
function serialOptions(selected) {
  const seen = new Set();
  const opts = [`<option value=""${selected ? "" : " selected"}>(auto)</option>`];
  DEVICES.forEach((d) => {
    seen.add(d.serial);
    const label = `${d.serial} · ${d.product}${d.node ? " (" + d.node + ")" : ""}`;
    opts.push(`<option value="${d.serial}"${d.serial === selected ? " selected" : ""}>${label}</option>`);
  });
  if (selected && !seen.has(selected))
    opts.push(`<option value="${selected}" selected>${selected} · (not detected)</option>`);
  return opts.join("");
}
function camRow(cam) {
  // role is assumed identical to the name for now, so the rail only edits
  // name / type / serial (gatherForm sends role = name).
  return `<div class="frow" data-cam>
    <input class="inp" data-k="name" value="${cam.name || ""}" placeholder="name" />
    <select class="sel" data-k="type" title="camera type">${optionTags(OPTS.camera_type, cam.type)}</select>
    <select class="sel" data-k="serial" title="device serial (from Detect)">${serialOptions(cam.serial || "")}</select>
    <button class="rm" data-rm title="remove row">×</button></div>`;
}

function renderRail(cfg) {
  OPTS = cfg.options;
  const robots = (cfg.robots || []).map(robotRow).join("");
  const ctrls = (cfg.controllers || []).map(controllerRow).join("");
  const cams = (cfg.cameras || []).map(camRow).join("");
  $("rail-body").innerHTML = `
    <datalist id="can-list">${optionTags(OPTS.can_channel || [], "")}</datalist>
    <div class="cfg"><h3>Robot</h3>
      <div class="frow head"><span>type</span><span>gripper</span><span>can bus</span><span></span></div>
      <div id="robot-rows">${robots}</div>
      <button class="add" id="add-robot">+ Add robot</button>
    </div>
    <div class="cfg"><h3>Controller</h3>
      <div class="frow head"><span>type</span><span>controls robot</span><span>can bus</span><span></span></div>
      <div id="ctrl-rows">${ctrls}</div>
      <button class="add" id="add-ctrl">+ Add controller</button>
    </div>
    <div class="cfg"><h3>Cameras</h3>
      <div class="frow head camhead"><span>name</span><span>type</span><span>serial</span><span></span></div>
      <div id="cam-rows">${cams}</div>
      <div class="cam-actions">
        <button class="add" id="add-cam">+ Add camera</button>
        <button class="add" id="redetect" title="rescan connected cameras">↻ Detect</button>
      </div>
    </div>
    <div class="cfg"><h3>Output</h3>
      <div class="cfg-row"><span class="k">format</span>
        <select class="sel" id="fmt">${optionTags(OPTS.data_format, cfg.data_format)}</select></div>
      <div class="cfg-row"><span class="k">save_root</span>
        <input class="inp" id="save-root" value="${cfg.save_root || "data/episodes"}" /></div>
    </div>
    <div class="cfg"><h3>Task</h3>
      <input class="inp" id="task" placeholder="task name (required to record)" required value="${cfg.task_name && cfg.task_name !== "unknown" && cfg.task_name !== "unknow" ? cfg.task_name : ""}" />
    </div>`;

  $("add-robot").onclick = () =>
    $("robot-rows").insertAdjacentHTML("beforeend", robotRow({ type: OPTS.robot_type[0], gripper: OPTS.gripper[0] }));
  $("add-ctrl").onclick = () =>
    $("ctrl-rows").insertAdjacentHTML("beforeend", controllerRow({ type: OPTS.controller_type[0], controls: OPTS.robot_type[0] }));
  $("add-cam").onclick = () =>
    $("cam-rows").insertAdjacentHTML("beforeend", camRow({ name: "", type: OPTS.camera_type[0].value }));
  $("redetect").onclick = redetect;
  // Row removal via delegation (rows are added dynamically).
  $("rail-body").addEventListener("click", (e) => {
    if (e.target.matches("[data-rm]")) e.target.closest(".frow").remove();
  });
}

// Collect the current rail state into the StationForm payload the API expects.
function gatherForm() {
  // A blank channel is sent as null, i.e. "derive from type" — not as "".
  const chan = (r) => r.querySelector("[data-k=channel]").value.trim() || null;
  const robots = [...document.querySelectorAll("#robot-rows [data-robot]")].map((r) => ({
    type: r.querySelector("[data-k=type]").value,
    gripper: r.querySelector("[data-k=gripper]").value,
    channel: chan(r),
  }));
  const controllers = [...document.querySelectorAll("#ctrl-rows [data-controller]")].map((r) => ({
    type: r.querySelector("[data-k=type]").value,
    controls: r.querySelector("[data-k=controls]").value,
    channel: chan(r),
  }));
  const cameras = [...document.querySelectorAll("#cam-rows [data-cam]")]
    .map((r) => {
      const name = r.querySelector("[data-k=name]").value.trim();
      return {
        name,
        type: r.querySelector("[data-k=type]").value,
        role: name, // assume role == name for now
        serial: r.querySelector("[data-k=serial]").value || null,
      };
    })
    .filter((c) => c.name);
  return {
    controllers,
    robots,
    cameras,
    data_format: $("fmt").value,
    save_root: $("save-root").value.trim() || "data/episodes",
    task_name: $("task").value.trim(),
  };
}

// --- camera grid (one tile per video stream; stereo => one tile per eye) ------
let camSig = null; // signature of the built grid; rebuild when the camera set changes
function camsSignature(cams) {
  return cams.map((c) => `${c.name}/${c.eye || "rgb"}`).join(",");
}
function buildCameras(cams) {
  camSig = camsSignature(cams);
  const stage = $("stage");
  stage.innerHTML = "";
  cams.forEach((c) => {
    const url = `/api/cameras/${c.name}/preview.jpg` + (c.eye ? `?eye=${c.eye}` : "");
    const el = document.createElement("div");
    el.className = "cam";
    el.innerHTML = `
      <div class="view">
        <img alt="${c.name}" />
        <div class="tag"><span class="role">${c.role}${c.eye ? "/" + c.eye : ""}</span><span class="nm">${c.name}</span></div>
        <div class="live"><i></i>LIVE</div>
      </div>
      <div class="foot"><span>${c.type}</span><span class="spec">${c.width}×${c.height} · ${c.fps}fps</span></div>`;
    el.querySelector("img").dataset.url = url;
    stage.appendChild(el);
  });
}
// Refresh preview images ~5 fps with a cache-buster.
setInterval(() => {
  document.querySelectorAll("#stage img").forEach((img) => {
    if (!img.dataset.url) return;
    img.src = img.dataset.url + (img.dataset.url.includes("?") ? "&" : "?") + "t=" + Date.now();
  });
}, 200);

// --- per-arm live telemetry --------------------------------------------------
// One card per driven arm from status.arms (see ControlLoop.joint_snapshot): that leader's
// own buttons/trigger, follower joints, action sent. Vectors are [arm_0..arm_{n-1}, gripper].
let armSig = "";

// Joint count from whichever vector has arrived; the YAM default until one does.
function armVecLen(arms) {
  for (const a of Object.values(arms || {})) {
    const v = a.follower || a.leader_raw || a.leader_cmd;
    if (v && v.length) return v.length;
  }
  return 7;
}

function armCard(name, nJoints) {
  const heads = Array.from({ length: nJoints }, (_, i) => `<span class="h">J${i}</span>`).join("");
  const cells = (row, extra = "") =>
    Array.from(
      { length: nJoints },
      (_, i) => `<span class="n ${extra}" data-${row}="${i}">—</span>`
    ).join("");
  // label column + one per arm joint + the gripper
  const cols = `grid-template-columns:2.4rem repeat(${nJoints + 1},1fr)`;
  return `<div class="armcard" data-arm="${name}">
    <div class="ahead">
      <span class="anm">${name}</span>
      <span class="ind" data-btn="0" title="leader top button (sync)"><i></i>TOP</span>
      <span class="ind" data-btn="1" title="leader second button (record)"><i></i>2ND</span>
      <div class="trig" title="leader trigger, normalized (0 closed → 1 open)"><span data-trigfill></span></div>
      <span class="n tnum" data-trigval>—</span>
    </div>
    <div class="jgrid" style="${cols}">
      <span></span>${heads}<span class="h">GRIP</span>
      <span class="rl" title="leader encoder angles straight off the bus (rad)">raw</span>${cells("raw")}<span class="n g" data-raw="g">—</span>
      <span class="rl seam" title="leader angles after this side's joint_signs (rad)">cal</span>${cells("cal", "seam")}<span class="n g seam" data-cal="g">—</span>
      <span class="rl" title="follower measured position">pos</span>${cells("pos")}<span class="n g" data-pos="g">—</span>
      <span class="rl act" title="action commanded to this arm">act</span>${cells("act")}<span class="n g" data-act="g">—</span>
    </div>
  </div>`;
}

// Fill one row from a [arm..., gripper] vector; a missing vector blanks the row ("act"
// while sync is off, raw/cal for a leader with no raw encoders). `normGrip` picks the
// gripper format: normalized 0..1 (pos/act) vs a trigger angle in rad (raw/cal).
function setArmRow(card, row, vec, normGrip) {
  card.querySelectorAll(`[data-${row}]`).forEach((el) => {
    const key = el.dataset[row];
    const i = key === "g" ? (vec ? vec.length - 1 : 0) : Number(key);
    const v = vec ? vec[i] : null;
    if (v == null) {
      el.textContent = "—";
    } else {
      // radians are signed for column alignment; a normalized gripper is 0..1
      el.textContent =
        key === "g" && normGrip
          ? Number(v).toFixed(2)
          : (v >= 0 ? "+" : "") + Number(v).toFixed(3);
    }
    el.classList.toggle("off", v == null);
  });
}

function renderArms(s) {
  const units = s.units || [];
  const n = armVecLen(s.arms);
  const sig = units.join(",") + "|" + n;
  if (sig !== armSig) {
    armSig = sig;
    $("armmon-grid").innerHTML = units.map((u) => armCard(u, n - 1)).join("");
  }
  $("armmon-hint").textContent = !units.length
    ? "no arms — Start Teleop"
    : s.estopped
    ? "e-stopped — nothing commanded"
    : s.teleop_running
    ? "synced — action is live"
    : "sync off — pos only, no action sent";
  for (const u of units) {
    const card = document.querySelector(`.armcard[data-arm="${u}"]`);
    if (!card) continue;
    const a = (s.arms || {})[u] || {};
    setArmRow(card, "raw", a.leader_raw, false);
    setArmRow(card, "cal", a.leader_cal, false);
    setArmRow(card, "pos", a.follower, true);
    setArmRow(card, "act", a.leader_cmd, true);
    const btns = a.leader_buttons || [];
    card.querySelectorAll("[data-btn]").forEach((el) =>
      el.classList.toggle("on", !!btns[Number(el.dataset.btn)]));
    const g = a.leader_grip;
    card.querySelector("[data-trigfill]").style.width = Math.round((Number(g) || 0) * 100) + "%";
    card.querySelector("[data-trigval]").textContent = g == null ? "—" : Number(g).toFixed(2);
  }
}

// --- websocket status feed ---------------------------------------------------
function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const s = JSON.parse(ev.data);
    const dot = $("conn-dot");
    dot.classList.toggle("bad", !!s.estopped);
    $("conn-txt").textContent = s.estopped ? "e-stopped" : "live";

    // GPU badge: mem used/total (pct) + util; goes red when memory is nearly full.
    if (s.gpu) {
      $("gpu-txt").textContent =
        `GPU ${(s.gpu.mem_used_mib / 1024).toFixed(1)}/${(s.gpu.mem_total_mib / 1024).toFixed(1)}G ` +
        `(${s.gpu.mem_pct}%) \u00b7 ${s.gpu.util_pct}%`;
      $("gpu-txt").style.color = s.gpu.mem_pct > 90 ? "#ff6b6b" : "";
    } else {
      $("gpu-txt").textContent = "GPU n/a";
    }

    // Arm/leader + error-summary badge: red ⚠ with the message on hover when a loop
    // crashed (teleop or deploy), amber when e-stopped, else the live unit count.
    const err = s.last_error || s.deploy_error || null;
    const nUnits = (s.units || []).length;
    const armTxt = $("arm-txt");
    if (err) {
      armTxt.textContent = "ARM ⚠";
      armTxt.style.color = "#ff6b6b";
      $("arm-box").title = err;
    } else if (s.estopped) {
      armTxt.textContent = `ARM ${nUnits} · e-stop`;
      armTxt.style.color = "#ffb020";
      $("arm-box").title = "e-stopped — Reset Session to recover, then Start Teleop";
    } else if (s.live) {
      armTxt.textContent = `ARM ${nUnits} · live`;
      armTxt.style.color = "";
      $("arm-box").title = (s.units || []).join(", ") || "live";
    } else {
      armTxt.textContent = nUnits ? `ARM ${nUnits}` : "ARM —";
      armTxt.style.color = "";
      $("arm-box").title = "idle — Start Teleop to go live";
    }

    // teleop button reflects sync state (top button / GUI drive the same flag)
    $("teleop-label").textContent = s.teleop_running ? "Stop Teleop" : "Start Teleop";
    $("btn-teleop").classList.toggle("ghost", s.teleop_running);
    $("btn-teleop").classList.toggle("primary", !s.teleop_running);
    // A deploy session runs on the follower buses only — there are no leaders to teleop
    // with and no teleop recorder, so both actions would just 409. Say why instead.
    const autonomy = !!s.followers_only;
    const autonomyWhy = "live for autonomy (follower buses only) — Stop the rollout and " +
      "Reset Session first";
    $("btn-teleop").disabled = autonomy;
    $("btn-teleop").title = autonomy ? autonomyWhy : "";
    // Recording is available once the system is live (devices up), not only when synced.
    $("btn-rec").disabled = !s.live || autonomy;
    $("btn-rec").title = autonomy ? autonomyWhy + " (the rollout records itself)" : "";

    // recording state
    const rec = !!s.recording;
    $("rec-label").textContent = rec ? "Stop Recording" : "Start Recording";
    $("rec-lamp").classList.toggle("on", rec);
    $("rec-lamp-txt").textContent = rec ? "recording" : "idle";
    $("stage").classList.toggle("rec", rec);

    // teaching-handle indicators: [top, second] buttons + trigger bar, merged across
    // arms (any leader fires sync/record). The per-arm panel shows each separately.
    const btn = s.buttons || [false, false];
    $("ind-top").classList.toggle("on", !!btn[0]);
    $("ind-second").classList.toggle("on", !!btn[1]);
    $("trig-fill").style.width = Math.round((Number(s.trigger) || 0) * 100) + "%";

    renderArms(s);

    // readouts
    $("eps").textContent = s.episodes_done;

    // (Re)build the camera grid when the camera set first arrives or changes.
    if (s.cameras && s.cameras.length && camsSignature(s.cameras) !== camSig)
      buildCameras(s.cameras);
    updateJobPills(s.jobs || []);
    // The in-session deploy client owns the deploy pill while a rollout runs, and
    // surfaces a rollout-thread crash (logged once) after it stops.
    if (s.deploying) {
      const r = s.deploy_recording ? ` · rec ${s.deploy_frames}f` : "";
      $("dp-status").textContent = `deploying · ${s.deploy_hz} Hz${r}`;
    } else if (s.deploy_error) {
      $("dp-status").textContent = `stopped · error: ${s.deploy_error}`;
      if (_dpErrShown !== s.deploy_error) {
        _dpErrShown = s.deploy_error;
        $("dp-logs").textContent += `[client] rollout stopped: ${s.deploy_error}\n`;
      }
    }
  };
  ws.onclose = () => {
    $("conn-dot").classList.add("bad");
    $("conn-txt").textContent = "reconnecting…";
    setTimeout(connectWS, 1000);
  };
}
function updateJobPills(jobs) {
  const ft = jobs.filter((j) => j.kind === "train").pop();
  const dp = jobs.filter((j) => j.kind === "deploy").pop();
  if (ft) {
    $("ft-status").textContent = `${ft.id}: ${ft.status}`;
    // Re-attach the loss-curve poller after a page reload / WS reconnect: the
    // in-memory _train.pts is gone but /metrics?since=0 backfills every point.
    // Idempotent — only (re)attach when the watched job id changes.
    if (_watchedTrain !== ft.id) {
      _watchedTrain = ft.id;
      $("ft-cmd").textContent = ft.real_command || $("ft-cmd").textContent;
      tailJob(ft.id, "ft-logs");
      watchTrain(ft.id);
    }
  }
  if (dp) $("dp-status").textContent = `${dp.id}: ${dp.status}`;
}

// --- collect controls --------------------------------------------------------
$("btn-teleop").onclick = async () => {
  // Toggle based on current label (status feed keeps it in sync afterwards).
  const running = $("teleop-label").textContent.startsWith("Stop");
  if (running) return post("/api/collect/stop-teleop");
  // Starting applies the current Station rail edits first; a rejected config (e.g. two
  // devices on one CAN bus) comes back as {detail}.
  const r = await post("/api/collect/start-teleop", gatherForm());
  if (r.detail) toast(`Start Teleop refused: ${r.detail}`, "err");
};
$("btn-rec").onclick = () => {
  const recording = $("rec-label").textContent.startsWith("Stop");
  if (recording) { post("/api/collect/stop-recording"); return; }
  const task = $("task").value.trim();
  if (!task) {
    // A nameless task poisons the dataset folder and the training prompt.
    alert("Set a Task name first \u2014 it becomes the episode folder and the training prompt\n(e.g. \"pick up the bottle\").");
    $("task").focus();
    return;
  }
  post("/api/collect/start-recording", { task_name: task });
};
// Lightweight toast so maintenance actions give visible feedback (they run
// server-side with no UI change otherwise -> users think the button is dead).
function toast(msg, kind) {
  let t = document.getElementById("fm-toast");
  if (!t) {
    t = document.createElement("div");
    t.id = "fm-toast";
    t.style.cssText = "position:fixed;bottom:20px;left:50%;transform:translateX(-50%);" +
      "z-index:200;padding:10px 16px;border-radius:8px;font-size:13px;color:#fff;" +
      "box-shadow:0 4px 16px rgba(0,0,0,.4);transition:opacity .3s;opacity:0";
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.style.background = kind === "err" ? "#cf222e" : (kind === "warn" ? "#9a6700" : "#1a7f37");
  t.style.opacity = "1";
  clearTimeout(t._h);
  t._h = setTimeout(() => { t.style.opacity = "0"; }, 3500);
}

$("estop").onclick = async () => {
  const r = await post("/api/estop");
  toast(`E-STOP sent — arms hold position (still powered). ` +
        `Reset Session then Start Teleop to resume.`, "warn");
};
$("power-off-arms").onclick = async () => {
  const ok = confirm(
    "Disable motor power on both follower arms?\n\n" +
    "Control will stop and the arms will lose holding torque. Support the arms before continuing."
  );
  if (!ok) return;
  const r = await post("/api/maintenance/power-off-arms");
  const failed = Object.entries(r.arms || {}).filter(([, v]) => Object.keys(v.failed || {}).length);
  if (r.ok) {
    toast("Motor power disabled — support the arms before moving them.", "warn");
  } else {
    toast(`Power-off incomplete: ${failed.map(([name]) => name).join(", ") || "no arm responded"}`, "err");
  }
};
$("end-hardware-session").onclick = async () => {
  const ok = confirm(
    "End the YAM-ABC-Reproduce hardware session?\n\n" +
    "This will E-STOP the arms, stop active deploy jobs, and close the YAM-ABC-Reproduce GUI service. " +
    "You must restart yam-abc-gui before using the robot again."
  );
  if (!ok) return;
  const result = await post("/api/maintenance/end-hardware-session");
  if (result.detail) {
    alert(`Could not end the session: ${result.detail}`);
    return;
  }
  alert("Hardware session ended. The YAM-ABC-Reproduce page will disconnect now.");
};
// Preview: (re)open cameras from the current rail config, without robot/teleop.
$("btn-preview").onclick = () => post("/api/collect/connect", gatherForm());
$("reset-can").onclick = async () => {
  toast("Resetting CAN buses\u2026");
  const r = await post("/api/maintenance/reset-can");
  toast(r && r.ok ? "CAN buses reset" : `Reset CAN: ${r && r.output ? r.output : "failed"}`,
        r && r.ok ? "ok" : "warn");
};
// Reset Session: recover after E-STOP / a failed start without restarting the GUI.
$("reset-session").onclick = async () => {
  if (!confirm("Reset the session? Stops the loop, releases the arms, and resets CAN. " +
               "Cameras stay open. Press Start Teleop afterwards to re-arm.")) return;
  const r = await post("/api/session/reset");
  toast("Session reset" + (r && r.can ? ` \u00b7 ${r.can}` : "") + " \u2014 press Start Teleop to re-arm.",
        r && r.ok === false ? "warn" : "ok");
};

// Zero a passive-GELLO leader: confirm (it writes EEPROM), post, report outcome.
async function zeroGello(side) {
  const ok = confirm(
    `Hold the ${side.toUpperCase()} GELLO at the follower's HOME pose, ` +
    `with the TRIGGER FULLY RELEASED, then confirm.\n\n` +
    `The trigger matters: its zero is the released position, and an off-centre ` +
    `gripper zero makes the gripper stick at one end.\n\n` +
    `This writes the leader's encoder EEPROM (re-zeroable anytime). ` +
    `Sync will be turned off first.`);
  if (!ok) return;
  const r = await post("/api/maintenance/zero-gello", { side });
  alert(r.ok ? `✓ ${r.message}` : `✗ zero failed: ${r.message}`);
}
$("zero-l").onclick = () => zeroGello("left");
$("zero-r").onclick = () => zeroGello("right");

// --- job log tailing ---------------------------------------------------------
function tailJob(jobId, preEl) {
  let cursor = 0;
  const el = $(preEl);
  el.textContent = "";
  const timer = setInterval(async () => {
    const r = await fetch(`/api/jobs/${jobId}/logs?since=${cursor}`).then((x) => x.json());
    if (r.lines && r.lines.length) {
      el.textContent += r.lines.join("\n") + "\n";
      el.scrollTop = el.scrollHeight;
      cursor = r.cursor;
    }
    const job = await fetch(`/api/jobs/${jobId}`).then((x) => x.json());
    if (job.status === "exited") clearInterval(timer);
  }, 400);
}

// --- train -------------------------------------------------------------------
let _train = { pts: [], steps: 0, t0: 0 };
let _watchedTrain = null; // id of the train job whose loss curve is being polled
const fmtEta = (s) => {
  s = Math.round(s);
  const h = Math.floor(s / 3600); // a full-corpus convert runs hours; "224m3s" is unreadable
  if (h) return `${h}h${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
  const m = Math.floor(s / 60);
  return m ? `${m}m${s % 60}s` : `${s}s`;
};

function drawLoss() {
  const cv = $("ft-plot");
  if (!cv) return;
  cv.width = cv.clientWidth || 900;
  const W = cv.width,
    H = cv.height,
    ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const padL = 46,
    padR = 12,
    padT = 12,
    padB = 20;
  ctx.strokeStyle = "#dce3ec";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, H - padB);
  ctx.lineTo(W - padR, H - padB);
  ctx.stroke();
  ctx.fillStyle = "#8492a6";
  ctx.font = "10px monospace";
  const pts = _train.pts.filter((p) => p.loss != null);
  if (!pts.length) {
    ctx.fillText("waiting for @metric lines…", padL + 30, H / 2);
    return;
  }
  const xs = pts.map((p) => p.step),
    ys = pts.map((p) => p.loss);
  const xmin = Math.min(...xs),
    xmax = Math.max(Math.max(...xs), _train.steps || 0) || 1;
  let ymin = Math.min(...ys),
    ymax = Math.max(...ys);
  if (ymin === ymax) {
    ymin -= 1;
    ymax += 1;
  }
  const X = (s) => padL + (W - padL - padR) * ((s - xmin) / Math.max(1, xmax - xmin));
  const Y = (v) => padT + (H - padT - padB) * (1 - (v - ymin) / (ymax - ymin));
  ctx.strokeStyle = "#2f70b7";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  pts.forEach((p, i) => {
    const px = X(p.step),
      py = Y(p.loss);
    i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  });
  ctx.stroke();
  ctx.fillText(ymax.toFixed(3), 4, padT + 8);
  ctx.fillText(ymin.toFixed(3), 4, H - padB);
  const last = pts[pts.length - 1];
  ctx.fillText(`loss ${last.loss}`, W - 120, padT + 10);
}

function watchTrain(jobId) {
  // Train fields are rendered dynamically as ftf-<name>; guard if absent.
  const stepsEl = $("ftf-steps");
  _train = { pts: [], steps: parseInt((stepsEl && stepsEl.value) || "0", 10), t0: Date.now() };
  drawLoss();
  let mcur = 0;
  const timer = setInterval(async () => {
    const m = await fetch(`/api/jobs/${jobId}/metrics?since=${mcur}`).then((r) => r.json());
    if (m.points && m.points.length) {
      _train.pts.push(...m.points);
      mcur = m.cursor;
      drawLoss();
      const done = _train.pts[_train.pts.length - 1].step || _train.pts.length;
      const total = _train.steps || done;
      $("ft-progfill").style.width = (Math.min(1, done / Math.max(1, total)) * 100).toFixed(1) + "%";
      const el = (Date.now() - _train.t0) / 1000,
        rate = done / Math.max(1e-6, el);
      const eta = rate > 0 ? Math.max(0, (total - done) / rate) : 0;
      $("ft-progress").textContent = `step ${done}/${total} · ${rate.toFixed(1)}/s · ETA ${fmtEta(eta)}`;
    }
    const job = await fetch(`/api/jobs/${jobId}`).then((r) => r.json());
    $("ft-status").textContent = `${job.id}: ${job.status}`;
    if (job.status === "exited") clearInterval(timer);
  }, 500);
}

// Convert has no @metric stream, so it drives the same progress bar from the files the
// converter writes (/api/jobs/<id>/convert-progress). Frames rather than episodes: takes vary
// from tens to thousands of frames, so an episode count is only loosely tied to work done.
// The counters only move when an episode commits (minutes apart) while the write rate moves
// continuously -- which is what makes the job look alive between commits.
function watchConvert(jobId) {
  const timer = setInterval(async () => {
    const job = await fetch(`/api/jobs/${jobId}`).then((r) => r.json());
    $("ft-status").textContent = `${job.id}: ${job.status}`;
    const p = await fetch(`/api/jobs/${jobId}/convert-progress`).then((r) => r.json()).catch(() => null);
    if (p && !p.detail) {
      $("ft-progfill").style.width = (p.frac * 100).toFixed(1) + "%";
      const bits = [`${p.episodes_done}/${p.episodes_total} episodes`];
      if (p.frames_total) bits.push(`${p.frames_done.toLocaleString()}/${p.frames_total.toLocaleString()} frames`);
      if (p.write_mib_s != null) bits.push(`${p.write_mib_s.toFixed(0)} MiB/s`);
      if (p.eta_s != null) bits.push(`ETA ${fmtEta(p.eta_s)}`);
      $("ft-progress").textContent = bits.join(" · ");
    }
    if (job.status === "exited") {
      clearInterval(timer);
      if (p && !p.detail && job.returncode === 0) $("ft-progfill").style.width = "100%";
    }
  }, 1000);
}

// Render the fields for the selected policy (fetched from /api/train/fields),
// so the form adapts per backend (pi0 / molmoact2 / abc).
async function renderTrainFields(backend) {
  const box = $("ft-fields");
  if (!box) return;
  let spec;
  try {
    spec = await fetch(`/api/train/fields?backend=${backend}`).then((r) => r.json());
  } catch {
    box.innerHTML = "<span class='muted'>failed to load fields</span>";
    return;
  }
  box.innerHTML = (spec.fields || [])
    .map((f) => {
      const id = `ftf-${f.name}`;
      if (f.type === "select") {
        const opts = (f.options || [])
          .map((o) => `<option ${o === f.default ? "selected" : ""}>${o}</option>`)
          .join("");
        return `<label class="fld"><span>${f.label}</span><select id="${id}" data-name="${f.name}">${opts}</select></label>`;
      }
      const t = f.type === "number" ? "number" : "text";
      const v = f.default ?? "";
      return `<label class="fld"><span>${f.label}</span><input id="${id}" data-name="${f.name}" type="${t}" value="${v}" /></label>`;
    })
    .join("");
  // Backend prerequisite (abc's prepared cache) — the one thing the form itself cannot show.
  const note = $("ft-note");
  if (note) {
    note.textContent = spec.note || "";
    note.hidden = !spec.note;
  }
}

function gatherTrainParams() {
  const params = { backend: $("ft-backend").value };
  $("ft-fields")
    .querySelectorAll("[data-name]")
    .forEach((el) => {
      params[el.dataset.name] = el.value;
    });
  return params;
}

$("ft-backend").addEventListener("change", () => renderTrainFields($("ft-backend").value));
renderTrainFields($("ft-backend").value); // initial render

// Convert recorded episodes to a training dataset — as a job so progress streams here.
$("btn-convert").onclick = async () => {
  const task = $("cv-task").value.trim();
  if (!task) { $("ft-logs").textContent += "[convert] enter a task (folder under data/episodes) first\n"; return; }
  const job = await post("/api/jobs", { kind: "convert", params: {
    task, to: $("cv-fmt").value, repo_id: $("cv-repo").value.trim() || task } });
  if (job.detail) { $("ft-logs").textContent += `[convert] refused: ${job.detail}\n`; return; }
  $("ft-status").textContent = `${job.id}: ${job.status}`;
  $("ft-cmd").textContent = job.real_command || "";
  $("ft-progfill").style.width = "0%";
  $("ft-progress").textContent = "";
  tailJob(job.id, "ft-logs");
  watchConvert(job.id);
};

$("btn-train").onclick = async () => {
  const job = await post("/api/jobs", { kind: "train", params: gatherTrainParams() });
  $("ft-cmd").textContent = job.real_command || "";
  _watchedTrain = job.id; // claim it so updateJobPills won't double-attach
  if (job.detail) { // surface backend refusals (bad dataset, busy GPU, ...)
    $("ft-status").textContent = "refused";
    $("ft-logs").textContent += `[client] Start refused: ${job.detail}\n`;
    return;
  }
  tailJob(job.id, "ft-logs");
  watchTrain(job.id);
};

// --- deploy ------------------------------------------------------------------
// Default port per backend; only the openpi backends (pi0 / pi0.5) use the config field.
const _DP_PORT = { pi0: 8000, pi05: 8001, molmoact2: 8202, abc: 8300 };
let _dpErrShown = null; // last rollout error logged, so the WS feed logs it once
$("dp-backend").addEventListener("change", () => {
  const b = $("dp-backend").value;
  $("dp-port").value = _DP_PORT[b] || 8000;
  // The config field is openpi-only: fill its per-backend default and disable it for
  // molmoact2/abc (which ignore it) so they don't show a stale pi0 config.
  const _OPENPI_CFG = { pi0: "pi0_yam_lora", pi05: "pi05_yam_lora" };
  $("dp-config").disabled = !(b in _OPENPI_CFG);
  $("dp-config").value = _OPENPI_CFG[b] || "";
  // Rollouts land in data/rollouts/<policy>/<task>/<ep> (recorder adds the task level).
  $("dp-save").value = `data/rollouts/${b}`;
});

// Same-machine: launch the policy SERVER locally as a Job (tails server logs).
// Remote GPU box: skip this, start the server there, and just Load & Run.
$("btn-serve").onclick = async () => {
  const backend = $("dp-backend").value;
  const ckpt = $("dp-ckpt").value.trim();
  // molmoact2 falls back to a default HF checkpoint; the others need an explicit path.
  // Catch an empty path here — otherwise the server dies deep inside with a cryptic
  // `FileNotFoundError: _METADATA` pointing at the wrong dir.
  if (!ckpt && backend !== "molmoact2") {
    $("dp-status").textContent = "set a checkpoint path first";
    $("dp-logs").textContent += "[client] Start Server: checkpoint path is empty — " +
      "enter the path on the server, e.g. checkpoints/<config>/<run>/<step>\n";
    return;
  }
  const job = await post("/api/jobs", {
    kind: "deploy",
    params: {
      backend,
      checkpoint: ckpt,
      config: $("dp-config").value,
      prompt: $("dp-prompt").value,
      port: Number($("dp-port").value),
    },
  });
  if (job.detail) { // 409 from the pre-start guard (port busy / low VRAM)
    $("dp-status").textContent = "refused";
    $("dp-logs").textContent += `[client] Start Server refused: ${job.detail}\n`;
    return;
  }
  $("dp-status").textContent = `${job.id}: ${job.status}`;
  tailJob(job.id, "dp-logs");
};

// Start the in-session client (shares the preview cameras + robots) against host:port.
$("btn-deploy").onclick = async () => {
  _dpErrShown = null; // let a fresh run surface its own error
  $("dp-status").textContent = "starting client…";
  // Optional home pose: 14 comma/space-separated joint values to move to first.
  const homeStr = $("dp-home").value.trim();
  const home = homeStr ? homeStr.split(/[\s,]+/).map(Number).filter((x) => !Number.isNaN(x)) : null;
  const r = await post("/api/deploy/start", {
    host: $("dp-host").value,
    port: Number($("dp-port").value),
    prompt: $("dp-prompt").value,
    record: $("dp-record").checked,
    save_root: $("dp-save").value,
    home_pose: home && home.length ? home : null,
    rtc: $("dp-rtc").checked,
    max_joint_speed: Number($("dp-clamp").value) || 0,
  });
  // FastAPI returns {detail: "..."} on error (400); surface it instead of failing silently.
  if (r.detail) {
    $("dp-status").textContent = `error: ${r.detail}`;
    $("dp-logs").textContent += `[client] start_deploy failed: ${r.detail}\n`; // persists
  } else {
    $("dp-status").textContent = `deploying${r.recording ? " · recording" : ""}`;
  }
};

$("btn-deploy-stop").onclick = async () => {
  const r = await post("/api/deploy/stop");
  const s = r.saved && r.saved.path ? ` · saved ${r.saved.frames}f → ${r.saved.path}` : "";
  $("dp-status").textContent = `stopped${s}`;
};
$("dp-estop").onclick = () => post("/api/estop");

// --- camera health: topbar badge + popup (status / holders / reset) ----------
const _camPop = document.createElement("div");
_camPop.style.cssText = "display:none;position:fixed;top:52px;right:12px;z-index:99;" +
  "background:#1c2128;color:#e6edf3;border:1px solid #444c56;border-radius:8px;" +
  "padding:10px;max-width:660px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.5)";
document.body.appendChild(_camPop);

function _camRow(c) {
  const det = c.detected === null ? "" : (c.detected ? "detected" : "NOT DETECTED");
  const st = c.streaming ? "streaming" : "no stream";
  const bad = !c.streaming || c.detected === false;
  return `<div style="display:flex;gap:8px;margin:3px 0;color:${bad ? "#ff6b6b" : "#7ee787"}">` +
    `<b>${c.name}</b><span>${c.type}</span><code>${c.serial || ""}</code>` +
    `<span>${det}</span><span>${st}</span></div>`;
}

async function refreshCamHealth(renderPop) {
  try {
    const wantFull = renderPop || _camPop.style.display !== "none";
    const d = await fetch(`/api/cameras/health${wantFull ? "?full=1" : ""}`).then((r) => r.json());
    const ok = d.cameras.filter((c) => c.streaming).length;
    $("cam-txt").textContent = `CAM ${ok}/${d.cameras.length}`;
    $("cam-txt").style.color = ok < d.cameras.length ? "#ff6b6b" : "";
    if (!renderPop && _camPop.style.display === "none") return;
    const holders = (d.holders || []).map((h) =>
      `<div style="display:flex;gap:8px;align-items:center;margin:3px 0">` +
      `<code>${h.pid}</code><span>${h.user}</span>` +
      `<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${h.cmd}">` +
      `${h.self ? "(this GUI) " : ""}${h.cmd || "?"} \u2014 ${h.devs.join(",")}</span>` +
      (h.self ? "" : `<button class="btn" data-camkill="${h.pid}">kill</button>`) + `</div>`
    ).join("") || "<em>no /dev/video holders</em>";
    _camPop.innerHTML =
      `<div style="margin-bottom:6px"><b>cameras</b></div>` + d.cameras.map(_camRow).join("") +
      `<div style="margin:8px 0 6px"><b>/dev/video holders</b></div>` + holders +
      `<div style="margin-top:8px"><button class="btn" id="btn-cam-reset">Reset Cameras (hw-reset + reopen)</button>` +
      `<span id="cam-reset-msg" style="margin-left:8px"></span></div>`;
    _camPop.querySelectorAll("[data-camkill]").forEach((b) => {
      b.onclick = async () => {
        if (!confirm(`Kill pid ${b.dataset.camkill}?`)) return;
        const r = await post(`/api/cameras/kill-holder?pid=${b.dataset.camkill}`);
        if (r.detail) alert(`failed: ${r.detail}`);
        refreshCamHealth(true);
      };
    });
    const rb = _camPop.querySelector("#btn-cam-reset");
    if (rb) rb.onclick = async () => {
      _camPop.querySelector("#cam-reset-msg").textContent = "resetting\u2026 (~10 s)";
      const r = await post("/api/cameras/reset");
      _camPop.querySelector("#cam-reset-msg").textContent =
        r.detail ? `refused: ${r.detail}` :
        `reset ${r.reset_devices} devices; streaming: ${(r.streaming || []).join(", ") || "none"}` +
        (r.error ? ` (reopen error: ${r.error})` : "");
      refreshCamHealth(true);
    };
  } catch (e) { /* ignore */ }
}
setInterval(() => refreshCamHealth(false), 5000);
refreshCamHealth(false);
{
  const box = $("cam-txt").parentElement;
  box.style.cursor = "pointer";
  box.onclick = () => {
    const show = _camPop.style.display === "none";
    _camPop.style.display = show ? "block" : "none";
    if (show) refreshCamHealth(true);
  };
}

// --- checkpoint dropdown: auto-scan trained checkpoints for the backend ------
async function loadCkpts(replace) {
  const b = $("dp-backend").value;
  try {
    const d = await fetch(`/api/deploy/checkpoints?backend=${encodeURIComponent(b)}`)
      .then((r) => r.json());
    const list = d.checkpoints || [];
    // d.note (e.g. molmoact2 "convert to HF first") is shown as each option's label
    // and as the field tooltip, so the caveat is visible in the dropdown.
    const lbl = d.note ? ` label="${d.note}"` : "";
    $("dp-ckpt-list").innerHTML = list.map((c) => `<option value="${c}"${lbl}></option>`).join("");
    $("dp-ckpt").title = d.note || "";
    if (d.note && list.length) $("dp-logs").textContent += `[client] note: ${d.note}\n`;
    const cur = $("dp-ckpt").value.trim();
    // Replace the value when: switching backend (stale path from another backend
    // loads the wrong architecture), or the field is empty.
    if ((replace || !cur || !list.includes(cur)) && list.length) {
      $("dp-ckpt").value = list[0];
    } else if (replace && !list.length) {
      $("dp-ckpt").value = "";
    }
  } catch (e) { /* ignore */ }
}
$("dp-backend").addEventListener("change", () => loadCkpts(true));
$("dp-ckpt").addEventListener("focus", () => loadCkpts(false));
loadCkpts(false);

// --- GPU process popup: click the topbar GPU badge to inspect / kill ---------
const _gpuPop = document.createElement("div");
_gpuPop.id = "gpu-pop";
_gpuPop.style.cssText = "display:none;position:fixed;top:52px;right:12px;z-index:99;" +
  "background:#1c2128;color:#e6edf3;border:1px solid #444c56;border-radius:8px;" +
  "padding:10px;max-width:640px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.5)";
document.body.appendChild(_gpuPop);

async function refreshGpuPop() {
  const d = await fetch("/api/gpu/procs").then((r) => r.json());
  if (!d.procs.length) {
    _gpuPop.innerHTML = "<em>no processes on the GPU</em>";
    return;
  }
  _gpuPop.innerHTML = d.procs.map((p) =>
    `<div style="display:flex;gap:8px;align-items:center;margin:3px 0">` +
    `<code>${p.pid}</code><span>${p.user}</span><b>${(p.mem_mib / 1024).toFixed(1)}G</b>` +
    `<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${p.cmd}">${p.cmd || "?"}</span>` +
    `<button class="btn" data-kill="${p.pid}"${p.own ? "" : ' title="other user \u2014 needs sudo on the server"'}>kill</button></div>`
  ).join("");
  _gpuPop.querySelectorAll("[data-kill]").forEach((b) => {
    b.onclick = async () => {
      const pid = b.dataset.kill;
      if (!confirm(`Kill pid ${pid}? This stops that process immediately.`)) return;
      const r = await post(`/api/gpu/kill?pid=${pid}`);
      if (r.detail) alert(`failed: ${r.detail}`);
      refreshGpuPop();
    };
  });
}
{
  const box = $("gpu-txt").parentElement;
  box.style.cursor = "pointer";
  box.onclick = () => {
    const show = _gpuPop.style.display === "none";
    _gpuPop.style.display = show ? "block" : "none";
    if (show) refreshGpuPop();
  };
}

// --- policy-server lifecycle: status badge + stop button ---------------------
let _srvReady = false;
async function pollServer() {
  const host = ($("dp-host").value || "127.0.0.1").trim();
  const port = Number($("dp-port").value) || 8000;
  try {
    const s = await fetch(`/api/deploy/server-status?host=${encodeURIComponent(host)}&port=${port}`)
      .then((r) => r.json());
    _srvReady = s.state === "ready";
    const txt = { ready: `server ready :${port}`, loading: "server loading\u2026",
                  error: "server error (see logs)", none: "no server" }[s.state] || s.state;
    const bg  = { ready: "#1a7f37", loading: "#9a6700", error: "#cf222e", none: "#57606a" }[s.state] || "#57606a";
    const b = $("srv-badge");
    b.textContent = txt; b.style.background = bg;
    b.title = s.pid ? `pid ${s.pid}: ${s.cmd || ""}` : "";
    $("btn-deploy").disabled = !_srvReady;
    $("btn-deploy").title = _srvReady ? "" : "start (or connect to) a policy server first";
  } catch (e) { /* GUI unreachable; leave badge as-is */ }
}
setInterval(pollServer, 3000);
pollServer();

$("btn-serve-stop").onclick = async () => {
  const port = Number($("dp-port").value) || 8000;
  const r = await post(`/api/deploy/server-stop?port=${port}`);
  $("dp-logs").textContent += `[client] Stop Server: jobs stopped=${r.stopped_jobs ?? "?"}, killed pids=${JSON.stringify(r.killed ?? [])}\n`;
  if (r.denied) {
    $("dp-logs").textContent += `[client] ${r.denied}\n`;
    $("dp-status").textContent = "can't stop server: " + r.denied;
  } else {
    $("dp-status").textContent = "server stopped";
  }
  pollServer();
};

// Rescan connected cameras (/api/cameras/detect) into DEVICES.
function loadDevices() {
  return fetch("/api/cameras/detect")
    .then((r) => r.json())
    .then((d) => { DEVICES = d.devices || []; })
    .catch(() => { DEVICES = []; });
}
// Re-scan and refresh the serial pickers in place, keeping current selections.
function redetect() {
  loadDevices().then(() => {
    document.querySelectorAll("#cam-rows [data-cam] [data-k=serial]").forEach((sel) => {
      sel.innerHTML = serialOptions(sel.value || "");
    });
  });
}

// --- boot --------------------------------------------------------------------
Promise.all([
  fetch("/api/config").then((r) => r.json()),
  fetch("/api/cameras").then((r) => r.json()),
  loadDevices(), // populate DEVICES before renderRail so serial pickers are filled
])
  .then(([cfg, cams]) => {
    renderRail(cfg);
    // Station default for the Deploy tab's home pose. Pre-filled, not forced: clearing
    // the field means "don't home" — the server applies no fallback of its own.
    if (cfg.deploy_home_pose && !$("dp-home").value.trim()) {
      $("dp-home").value = cfg.deploy_home_pose.join(", ");
    }
    buildCameras(cams.cameras);
    // Open the cameras for preview right away (no robot/teleop), so tiles go live
    // without clicking Start Teleop. The WS feed then (re)builds the grid.
    // Boot should use the persisted YAML mapping. Submitting the rendered rail
    // here can write stale browser selections back over cameras.yaml on refresh.
    post("/api/collect/connect").catch(() => {});
  })
  .catch(() => {});
connectWS();

// --- review (post-collection sanity check) -----------------------------------
(function () {
  const JOINT_COLORS = ["#2f70b7", "#e0574f", "#327a3f", "#9b5cd0", "#c9820e", "#0e9488"];
  let R = null;

  let _eps = []; // cached episode list (filtered client-side)

  function renderList() {
    const list = $("rv-list");
    const src = $("rv-source").value;
    const task = $("rv-task").value;
    const eps = _eps.filter((e) => (!src || e.source === src) && (!task || e.task === task));
    list.innerHTML = eps.length
      ? eps
          .map((e) => {
            const badge = `<span class='ep-badge ${e.source}'>${e.source}</span>`;
            const pol = e.policy ? ` <span class='ep-pol'>${e.policy}</span>` : "";
            return `<div class='ep-row'><button class='ep-item' data-id='${e.id}'><div class='ep-task'>${badge}${pol} ${
              e.task || e.task_name || "(untitled)"
            }</div><div class='ep-sub tnum'>${e.num_frames ?? "?"}f · ${(e.cameras || []).join(
              "/"
            )}</div><div class='ep-sub'>${e.created_at || e.id}</div></button><button class='ep-del' data-id='${
              e.id
            }' title='delete episode'>✕</button></div>`;
          })
          .join("")
      : "<div class='muted'>no episodes match</div>";
    list.querySelectorAll(".ep-item").forEach((b) =>
      b.addEventListener("click", () => {
        list.querySelectorAll(".ep-item").forEach((x) => x.classList.remove("on"));
        b.classList.add("on");
        openEpisode(b.dataset.id);
      })
    );
    list.querySelectorAll(".ep-del").forEach((b) =>
      b.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const id = b.dataset.id;
        if (!confirm(`Delete episode ${id}? This cannot be undone.`)) return;
        await fetch(`/api/review/${id}`, { method: "DELETE" });
        if (R && R.id === id) {
          R = null;
          $("rv-head").textContent = "Select an episode to review";
          $("rv-videos").innerHTML = "";
        }
        loadList();
      })
    );
  }

  async function loadList() {
    const list = $("rv-list");
    list.innerHTML = "<div class='muted'>loading…</div>";
    try {
      _eps = (await fetch("/api/review/episodes").then((r) => r.json())).episodes || [];
      // Populate the task filter from the current episodes (keep the selection).
      const tasks = [...new Set(_eps.map((e) => e.task).filter(Boolean))].sort();
      const sel = $("rv-task");
      const cur = sel.value;
      sel.innerHTML =
        "<option value=''>all tasks</option>" +
        tasks.map((t) => `<option value='${t}'>${t}</option>`).join("");
      if (tasks.includes(cur)) sel.value = cur;
      renderList();
    } catch {
      list.innerHTML = "<div class='muted'>failed to load</div>";
    }
  }
  $("rv-source").addEventListener("change", renderList);
  $("rv-task").addEventListener("change", renderList);

  async function openEpisode(id) {
    const [meta, sig] = await Promise.all([
      fetch(`/api/review/${id}/meta`).then((r) => r.json()),
      fetch(`/api/review/${id}/signals`).then((r) => r.json()),
    ]);
    const fps = meta.control_hz || 30;
    R = { id, meta, sig, fps, T: sig.num_frames || 0, frame: 0, playing: false, videos: [] };
    $("rv-head").textContent = `${meta.task_name || "(untitled)"} — ${R.T} frames @ ${fps} Hz`;
    const vwrap = $("rv-videos");
    vwrap.innerHTML = "";
    R.videos = [];
    (meta.cameras || []).forEach((c) => {
      const box = document.createElement("div");
      box.className = "vid";
      const v = document.createElement("video");
      v.src = `/api/review/${id}/video/${c.role}`;
      v.muted = true;
      v.preload = "auto";
      v.playsInline = true;
      box.appendChild(v);
      const cap = document.createElement("div");
      cap.className = "vid-cap";
      cap.textContent = c.role;
      box.appendChild(cap);
      vwrap.appendChild(box);
      R.videos.push(v);
    });
    const armSel = $("rv-arm");
    armSel.innerHTML = (sig.arm_names || []).map((a) => `<option>${a}</option>`).join("");
    armSel.onchange = drawPlot;
    const scrub = $("rv-scrub");
    scrub.max = Math.max(0, R.T - 1);
    scrub.value = 0;
    scrub.oninput = () => {
      pause();
      seekFrame(parseInt(scrub.value, 10));
    };
    $("rv-play").onclick = () => (R.playing ? pause() : play());
    flags(sig, meta);
    seekFrame(0);
  }

  function seekFrame(f) {
    if (!R) return;
    R.frame = Math.max(0, Math.min(R.T - 1, f));
    const t = R.frame / R.fps;
    R.videos.forEach((v) => {
      try {
        v.currentTime = t;
      } catch {}
    });
    $("rv-scrub").value = R.frame;
    $("rv-frame").textContent = `${R.frame} / ${Math.max(0, R.T - 1)}`;
    drawPlot();
  }

  function play() {
    if (!R) return;
    R.playing = true;
    $("rv-play").textContent = "Pause";
    R.videos.forEach((v) => v.play().catch(() => {}));
    const step = () => {
      if (!R || !R.playing) return;
      const m = R.videos[0];
      if (m) {
        R.frame = Math.max(0, Math.min(R.T - 1, Math.round(m.currentTime * R.fps)));
        for (let i = 1; i < R.videos.length; i++) {
          const o = R.videos[i];
          if (Math.abs(o.currentTime - m.currentTime) > 0.08) {
            try {
              o.currentTime = m.currentTime;
            } catch {}
          }
        }
        $("rv-scrub").value = R.frame;
        $("rv-frame").textContent = `${R.frame} / ${Math.max(0, R.T - 1)}`;
        drawPlot();
        if (R.frame >= R.T - 1) return pause();
      }
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  function pause() {
    if (!R) return;
    R.playing = false;
    $("rv-play").textContent = "Play";
    R.videos.forEach((v) => v.pause());
  }

  function drawPlot() {
    const cv = $("rv-plot");
    if (!cv || !R) return;
    cv.width = cv.clientWidth || 900;
    const W = cv.width,
      H = cv.height,
      ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    const arm = $("rv-arm").value || (R.sig.arm_names || [])[0];
    const A = (R.sig.arms || {})[arm];
    if (!A || !A.action_joint || !A.action_joint.length) return;
    const aj = A.action_joint,
      ag = A.action_gripper,
      T = R.T;
    const padL = 34,
      padR = 10,
      padT = 12,
      padB = 14;
    const jH = (H - padT - padB) * 0.68,
      gY0 = padT + jH + 12,
      gH = H - padB - gY0;
    let lo = Infinity,
      hi = -Infinity;
    for (const row of aj) for (const v of row) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (lo === hi) {
      lo -= 1;
      hi += 1;
    }
    const x = (i) => padL + (W - padL - padR) * (T <= 1 ? 0 : i / (T - 1));
    const yj = (v) => padT + jH * (1 - (v - lo) / (hi - lo));
    const yg = (v) => gY0 + gH * (1 - Math.max(0, Math.min(1, v)));
    ctx.strokeStyle = "#dce3ec";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, H - padB);
    ctx.stroke();
    for (let j = 0; j < aj[0].length; j++) {
      ctx.strokeStyle = JOINT_COLORS[j % JOINT_COLORS.length];
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let i = 0; i < T; i++) {
        const px = x(i),
          py = yj(aj[i][j]);
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.stroke();
    }
    if (ag && ag.length) {
      ctx.strokeStyle = "#172033";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let i = 0; i < T; i++) {
        const g = Array.isArray(ag[i]) ? ag[i][0] : ag[i];
        const px = x(i),
          py = yg(g);
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.stroke();
    }
    ctx.fillStyle = "#8492a6";
    ctx.font = "10px monospace";
    ctx.fillText("action joints (rad)", padL + 4, padT + 10);
    ctx.fillText("gripper [0,1]", padL + 4, gY0 + 10);
    const phx = x(R.frame);
    ctx.strokeStyle = "#e0574f";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(phx, padT);
    ctx.lineTo(phx, H - padB);
    ctx.stroke();
  }

  function flags(sig, meta) {
    const el = $("rv-flags"),
      T = sig.num_frames || 0,
      msgs = [];
    for (const [role, c] of Object.entries(sig.cameras || {})) {
      const n = c.timestamps ? c.timestamps.length : 0;
      if (n !== T) msgs.push(`camera ${role}: ${n} frames vs ${T}`);
      if (c.timestamps && c.timestamps.length > 2) {
        let mx = 0;
        for (let i = 1; i < c.timestamps.length; i++)
          mx = Math.max(mx, c.timestamps[i] - c.timestamps[i - 1]);
        const exp = 1000 / (meta.control_hz || 30);
        if (mx > exp * 2.5) msgs.push(`${role}: max frame gap ${mx.toFixed(0)}ms (~${exp.toFixed(0)}ms expected)`);
      }
    }
    el.innerHTML = msgs.length
      ? msgs.map((m) => `<span class='flag warn'>⚠ ${m}</span>`).join("")
      : `<span class='flag ok'>✓ ${T} frames, cameras aligned</span>`;
  }

  document.addEventListener("click", (e) => {
    if (e.target && e.target.id === "rv-refresh") loadList();
  });
  window.addEventListener("load", () => {
    const cv = $("rv-plot");
    if (cv)
      cv.addEventListener("click", (e) => {
        if (!R) return;
        const r = cv.getBoundingClientRect();
        pause();
        seekFrame(Math.round(((e.clientX - r.left) / r.width) * (R.T - 1)));
      });
  });
  window.addEventListener("resize", () => R && drawPlot());
  let loaded = false;
  document.querySelectorAll(".tabs button").forEach((b) =>
    b.addEventListener("click", () => {
      if (b.dataset.tab === "review" && !loaded) {
        loaded = true;
        loadList();
      }
    })
  );
})();
