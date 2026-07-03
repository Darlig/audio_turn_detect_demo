import {
  Room,
  RoomEvent,
  Track,
  createLocalAudioTrack,
} from "https://cdn.jsdelivr.net/npm/livekit-client@2.15.5/dist/livekit-client.esm.mjs";

const TRANSCRIPTION_TOPIC = "lk.transcription";
const TRANSCRIPTION_FINAL_ATTR = "lk.transcription_final";
const TRANSCRIPTION_SEGMENT_ATTR = "lk.segment_id";
const AGENT_EOU_DEBUG_TOPIC = "audio-turn-demo.agent-eou-debug";
const AGENT_LATENCY_METRICS_TOPIC = "audio-turn-demo.latency-metrics";
const REPLAY_LEAD_IN_MS = 800;
const REPLAY_TAIL_MS = 1800;
const TRANSCRIPT_DEDUPE_MS = 15000;
const FINAL_BUBBLE_SETTLE_MS = 1800;
const ASSISTANT_READY_SETTLE_MS = 2200;
const PENDING_AGENT_IDENTITY = "__pending_agent_dispatch__";

const state = {
  room: null,
  micTrack: null,
  fileTrack: null,
  fileAudio: null,
  fileContext: null,
  fileSilenceSource: null,
  fileStopTimer: null,
  debugSimTrack: null,
  debugSimContext: null,
  debugSimDestination: null,
  debugSimSilenceSource: null,
  debugSimAudio: null,
  debugSimAudioSource: null,
  debugSimAudioUrl: "",
  debugTtsPlaying: false,
  points: [],
  threshold: null,
  pointTimeOffset: null,
  connected: false,
  connecting: false,
  activeInput: null,
  stoppedSinceLastRun: false,
  activeTrackSid: null,
  acceptingPoints: false,
  remoteAudioElements: new Map(),
  agentDispatchRequested: false,
  agentDispatchStartedAt: 0,
  knownAgentIdentities: new Set(),
  expectedAgentIdentity: null,
  activeAgentIdentity: null,
  audioContext: null,
  audioUnlocked: false,
  agentAudioElement: null,
  debugTtsSynthesizing: false,
  debugTtsBlob: null,
  debugTtsPreparedText: "",
  debugTtsPaused: false,
  debugTtsEnded: false,
  clientIdentity: null,
  transcriptSeen: new Map(),
  transcriptContentSeen: new Map(),
  liveBubbles: {
    user: null,
    assistant: null,
  },
  liveBubbleTimers: {
    user: null,
    assistant: null,
  },
  assistantReadyTimer: null,
};

const el = {
  badge: document.getElementById("connectionBadge"),
  roomName: document.getElementById("roomName"),
  connectBtn: document.getElementById("connectBtn"),
  micBtn: document.getElementById("micBtn"),
  fileButton: document.getElementById("fileButton"),
  fileInput: document.getElementById("fileInput"),
  debugTtsText: document.getElementById("debugTtsText"),
  debugTtsSessionBtn: document.getElementById("debugTtsSessionBtn"),
  debugTtsBtn: document.getElementById("debugTtsBtn"),
  debugTtsStartBtn: document.getElementById("debugTtsStartBtn"),
  debugTtsPauseBtn: document.getElementById("debugTtsPauseBtn"),
  debugTtsEndBtn: document.getElementById("debugTtsEndBtn"),
  debugTtsStatus: document.getElementById("debugTtsStatus"),
  agentEouStatus: document.getElementById("agentEouStatus"),
  agentEouProbability: document.getElementById("agentEouProbability"),
  agentEouThreshold: document.getElementById("agentEouThreshold"),
  agentEouDelay: document.getElementById("agentEouDelay"),
  agentEouResult: document.getElementById("agentEouResult"),
  agentEouTrigger: document.getElementById("agentEouTrigger"),
  agentEouSource: document.getElementById("agentEouSource"),
  latencyStatus: document.getElementById("latencyStatus"),
  latencyStt: document.getElementById("latencyStt"),
  latencyEouWait: document.getElementById("latencyEouWait"),
  latencyLlm: document.getElementById("latencyLlm"),
  latencyTts: document.getElementById("latencyTts"),
  latencyTotal: document.getElementById("latencyTotal"),
  latencyMode: document.getElementById("latencyMode"),
  stopBtn: document.getElementById("stopBtn"),
  score: document.getElementById("scoreValue"),
  threshold: document.getElementById("thresholdValue"),
  decision: document.getElementById("decisionValue"),
  time: document.getElementById("timeValue"),
  subtitle: document.getElementById("sessionSubtitle"),
  dialogueState: document.getElementById("dialogueState"),
  agentState: document.getElementById("agentState"),
  inputState: document.getElementById("inputState"),
  audioState: document.getElementById("audioState"),
  userTranscript: document.getElementById("userTranscript"),
  userTranscriptStatus: document.getElementById("userTranscriptStatus"),
  assistantTranscript: document.getElementById("assistantTranscript"),
  assistantTranscriptStatus: document.getElementById("assistantTranscriptStatus"),
  conversationFeed: document.getElementById("conversationFeed"),
  conversationStatus: document.getElementById("conversationStatus"),
  log: document.getElementById("log"),
  canvas: document.getElementById("timeline"),
  remoteAudio: document.getElementById("remoteAudio"),
  remoteAudioStatus: document.getElementById("remoteAudioStatus"),
};

const ctx = el.canvas.getContext("2d");

function setBadge(text, mode = "") {
  el.badge.textContent = text;
  el.badge.className = `badge ${mode}`.trim();
}

function log(message) {
  const line = `[${new Date().toLocaleTimeString()}] ${message}`;
  el.log.textContent = `${line}\n${el.log.textContent}`.slice(0, 6000);
}

function setText(node, text, mode = "") {
  node.textContent = text;
  node.classList.remove("status-ok", "status-warn", "status-error", "status-active");
  if (mode) {
    node.classList.add(`status-${mode}`);
  }
}

function setDialogueState(text, mode = "") {
  setText(el.dialogueState, text, mode);
}

function setAgentState(text, mode = "") {
  setText(el.agentState, text, mode);
}

function setInputState(text, mode = "") {
  setText(el.inputState, text, mode);
}

function setAudioState(text, mode = "") {
  setText(el.audioState, text, mode);
  setText(el.remoteAudioStatus, text, mode);
}

function setControls() {
  const debugActive = state.activeInput === "debug-tts";
  const otherInputActive = Boolean(state.activeInput && !debugActive);
  const debugTextReady = Boolean(el.debugTtsText.value.trim());
  const debugAudioReady = hasPreparedDebugTts();
  el.connectBtn.disabled = state.connected || state.connecting;
  el.micBtn.disabled =
    !state.connected || state.connecting || Boolean(state.activeInput) || !canUseMicrophone();
  el.fileInput.disabled = !state.connected || state.connecting || Boolean(state.activeInput);
  el.fileButton.classList.toggle("disabled", el.fileInput.disabled);
  el.debugTtsSessionBtn.disabled =
    !state.connected ||
    state.connecting ||
    state.debugTtsSynthesizing ||
    otherInputActive ||
    debugActive;
  el.debugTtsBtn.disabled =
    !state.connected ||
    state.connecting ||
    otherInputActive ||
    state.debugTtsSynthesizing ||
    !debugTextReady;
  el.debugTtsStartBtn.textContent =
    debugActive && state.debugTtsPaused && !state.debugTtsEnded ? "Resume replay" : "Start replay";
  el.debugTtsStartBtn.disabled =
    !state.connected ||
    state.connecting ||
    state.debugTtsSynthesizing ||
    otherInputActive ||
    (debugActive
      ? state.debugTtsPlaying
        ? !state.debugTtsPaused
        : !debugAudioReady
      : !debugAudioReady);
  el.debugTtsPauseBtn.disabled =
    !debugActive || !state.debugTtsPlaying || state.debugTtsPaused || !state.debugSimAudio;
  el.debugTtsEndBtn.disabled = !debugActive;
  el.stopBtn.disabled = !state.activeInput;
  updateDebugTtsStatus();
}

function updateDebugTtsStatus() {
  if (state.debugTtsSynthesizing) {
    setText(el.debugTtsStatus, "synthesizing", "active");
    return;
  }
  if (state.activeInput === "debug-tts") {
    if (state.debugTtsPlaying && state.debugTtsPaused) {
      setText(el.debugTtsStatus, "holding silence", "warn");
    } else if (state.debugTtsPlaying) {
      setText(el.debugTtsStatus, "replaying", "active");
    } else if (state.debugTtsEnded) {
      setText(el.debugTtsStatus, "clip ended, holding silence", "warn");
    } else if (hasPreparedDebugTts()) {
      setText(el.debugTtsStatus, "holding silence, audio ready", "ok");
    } else if (state.debugTtsPaused) {
      setText(el.debugTtsStatus, "holding silence", "warn");
    } else {
      setText(el.debugTtsStatus, "holding silence", "warn");
    }
    return;
  }
  if (!state.connected) {
    setText(el.debugTtsStatus, "connect first");
    return;
  }
  if (state.activeInput) {
    setText(el.debugTtsStatus, "input active", "warn");
    return;
  }
  if (!el.debugTtsText.value.trim()) {
    setText(el.debugTtsStatus, "enter text", "warn");
    return;
  }
  if (hasPreparedDebugTts()) {
    setText(el.debugTtsStatus, "audio ready", "ok");
    return;
  }
  setText(el.debugTtsStatus, "synthesize first");
}

function restoreDebugTtsInputState() {
  if (state.activeInput !== "debug-tts") {
    setInputState("idle");
    return;
  }
  if (state.debugTtsPlaying && !state.debugTtsPaused) {
    setInputState("debug TTS replay", "active");
    return;
  }
  setInputState("debug TTS silence", "active");
}

function hasPreparedDebugTts() {
  return Boolean(
    state.debugTtsBlob &&
      state.debugTtsPreparedText &&
      state.debugTtsPreparedText === el.debugTtsText.value.trim()
  );
}

function canUseMicrophone() {
  return Boolean(window.isSecureContext && navigator.mediaDevices?.getUserMedia);
}

function microphoneUnavailableMessage() {
  if (!window.isSecureContext) {
    return "microphone requires HTTPS on LAN; use https://SERVER_IP:8090 or open from localhost";
  }
  return "microphone API is unavailable in this browser";
}

async function connectRoom() {
  if (state.connected || state.connecting) {
    return;
  }
  state.connecting = true;
  resetConversationView();
  setDialogueState("connecting", "active");
  setAgentState("not joined");
  setInputState("idle");
  setControls();
  const roomName = el.roomName.value.trim() || "audio-turn-demo";
  const identity = getClientIdentity();
  el.subtitle.textContent = roomName;
  setBadge("connecting");
  log("requesting LiveKit token");
  state.expectedAgentIdentity = PENDING_AGENT_IDENTITY;
  state.knownAgentIdentities.clear();
  const tokenParams = new URLSearchParams({ room: roomName, identity });
  const tokenResp = await fetch(`/token?${tokenParams.toString()}`);
  if (!tokenResp.ok) {
    throw new Error(await tokenResp.text());
  }
  const token = await tokenResp.json();
  log(`connecting to ${token.url}`);
  const room = new Room();
  registerTranscriptionStreamHandler(room);
  room.on("connected", () => {
    state.room = room;
    state.connected = true;
    state.connecting = false;
    setBadge("connected", "ok");
    setDialogueState("connected", "ok");
    setControls();
    log("room connected");
  });
  room.on("disconnected", () => {
    clearLiveBubbleTimer("user");
    clearLiveBubbleTimer("assistant");
    clearAssistantReadyTimer();
    state.connected = false;
    state.connecting = false;
    state.agentDispatchRequested = false;
    state.agentDispatchStartedAt = 0;
    state.knownAgentIdentities.clear();
    state.expectedAgentIdentity = null;
    state.activeAgentIdentity = null;
    clearRemoteAudio();
    setBadge("offline");
    setDialogueState("offline");
    setAgentState("not joined");
    setInputState("idle");
    setControls();
    log("room disconnected");
  });
  room.on(RoomEvent.ParticipantConnected, (participant) => {
    log(`participant joined: ${participant.identity}`);
    if (isAgentParticipant(participant) && acceptExpectedAgentParticipant(participant)) {
      attachParticipantAudioFromPublications(participant);
    } else if (isAgentParticipant(participant)) {
      if (!state.agentDispatchRequested) {
        state.knownAgentIdentities.add(participant.identity);
      }
      log(`ignored stale agent participant: ${participant.identity}`);
    }
  });
  room.on(RoomEvent.ParticipantDisconnected, (participant) => {
    log(`participant left: ${participant.identity}`);
    if (participant.identity === state.activeAgentIdentity) {
      state.activeAgentIdentity = null;
      setAgentState("left", "warn");
      setDialogueState("agent left", "warn");
    }
    detachParticipantAudio(participant);
  });
  room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
    log(`track subscribed: ${participant.identity} ${track.kind}`);
    if (isAudioTrack(track) && isAgentParticipant(participant) && acceptExpectedAgentParticipant(participant)) {
      attachRemoteAudio(track, publication, participant);
    } else if (isAudioTrack(track) && isAgentParticipant(participant)) {
      log(`ignored stale agent audio: ${participant.identity}`);
    }
  });
  room.on(RoomEvent.TrackUnsubscribed, (track, publication, participant) => {
    log(`track unsubscribed: ${participant.identity} ${track.kind}`);
    if (isAudioTrack(track)) {
      detachRemoteAudio(track, publication, participant);
    }
  });
  room.on(RoomEvent.DataReceived, (...args) => {
    const [payload, participant, _kind, topic] = args;
    handleRoomData(payload, participant, topic);
  });
  room.on(RoomEvent.TranscriptionReceived, (segments, participant) => {
    for (const segment of segments) {
      const text = segment.text?.trim();
      if (!text) {
        continue;
      }
      const speaker = participant?.identity || "unknown";
      const status = segment.final ? "final" : "interim";
      log(`transcript event ${status} ${speaker} segment=${segment.id || "--"}: ${text}`);
      handleTranscript(speaker, text, segment.final, segment.id);
    }
  });
  await withTimeout(room.connect(token.url, token.token), 15000, "LiveKit connection timed out");
  rememberExistingAgentParticipants(room);
  await startRoomAudio(room);
  state.room = room;
  state.connected = true;
  state.connecting = false;
  setBadge("connected", "ok");
  setDialogueState("waiting agent", "active");
  setControls();
  log(`joined room ${token.room} as ${token.identity}`);
  await dispatchAgent(token.room, token.identity);
}

function isAgentParticipant(participant) {
  return participant?.identity?.startsWith("agent-");
}

function isExpectedAgentParticipant(participant) {
  if (!isAgentParticipant(participant)) {
    return false;
  }
  if (state.expectedAgentIdentity === PENDING_AGENT_IDENTITY) {
    return false;
  }
  return !state.expectedAgentIdentity || participant.identity === state.expectedAgentIdentity;
}

function acceptExpectedAgentParticipant(participant) {
  if (!isAgentParticipant(participant)) {
    return false;
  }
  if (isExpectedAgentParticipant(participant)) {
    state.activeAgentIdentity = participant.identity;
    setAgentState("joined", "ok");
    setDialogueState(state.activeInput ? "listening" : "ready", "ok");
    return true;
  }
  if (
    state.expectedAgentIdentity === PENDING_AGENT_IDENTITY &&
    state.agentDispatchRequested &&
    !state.knownAgentIdentities.has(participant.identity)
  ) {
    acceptExpectedAgentIdentity(participant.identity);
    return true;
  }
  return false;
}

function acceptExpectedAgentIdentity(identity) {
  state.expectedAgentIdentity = identity;
  state.activeAgentIdentity = identity;
  setAgentState("joined", "ok");
  setDialogueState(state.activeInput ? "listening" : "ready", "ok");
  log(`using agent participant: ${identity}`);
}

function rememberExistingAgentParticipants(room) {
  state.knownAgentIdentities.clear();
  for (const participant of room.remoteParticipants?.values?.() || []) {
    if (isAgentParticipant(participant)) {
      state.knownAgentIdentities.add(participant.identity);
    }
  }
}

function isAudioTrack(track) {
  return track.kind === "audio" || track.kind === Track.Kind?.Audio;
}

function registerTranscriptionStreamHandler(room) {
  if (typeof room.registerTextStreamHandler !== "function") {
    log("LiveKit text stream transcription handler is unavailable in this client");
    return;
  }
  try {
    room.registerTextStreamHandler(TRANSCRIPTION_TOPIC, async (reader, participantInfo) => {
      const initialAttributes = reader.info?.attributes || {};
      const speaker = participantInfo?.identity || "unknown";
      const segmentId =
        initialAttributes[TRANSCRIPTION_SEGMENT_ATTR] ||
        reader.info?.id ||
        `${speaker}-${Date.now()}`;
      let text = "";

      try {
        for await (const chunk of reader) {
          text += chunk;
          const partial = text.trim();
          if (partial) {
            handleTranscript(speaker, partial, false, segmentId);
          }
        }
        const finalAttributes = reader.info?.attributes || initialAttributes;
        const isFinal = finalAttributes[TRANSCRIPTION_FINAL_ATTR] === "true";
        const finalText = text.trim();
        if (finalText) {
          handleTranscript(speaker, finalText, isFinal, segmentId);
        }
        log(
          `transcript stream ${isFinal ? "final" : "interim"} ${speaker} segment=${segmentId}: ${
            finalText || "--"
          }`
        );
      } catch (error) {
        log(`transcript stream failed: ${error.message}`);
      }
    });
  } catch (error) {
    log(`transcript stream handler unavailable: ${error.message}`);
  }
}

function handleRoomData(payload, participant, topic = "") {
  if (topic && topic !== AGENT_EOU_DEBUG_TOPIC && topic !== AGENT_LATENCY_METRICS_TOPIC) {
    return;
  }
  let message;
  try {
    message = JSON.parse(decodeRoomData(payload));
  } catch (_error) {
    return;
  }
  if (message?.type !== "agent_eou_debug" && message?.type !== "latency_metrics") {
    return;
  }
  const speaker = participant?.identity || "agent";
  if (
    state.activeAgentIdentity &&
    participant?.identity &&
    participant.identity !== state.activeAgentIdentity
  ) {
    log(`ignored stale agent data from ${speaker}`);
    return;
  }
  if (message.type === "agent_eou_debug") {
    updateAgentEouDebug(message, speaker);
  } else {
    updateLatencyMetrics(message, speaker);
  }
}

function decodeRoomData(payload) {
  if (typeof payload === "string") {
    return payload;
  }
  if (payload instanceof Uint8Array || payload instanceof ArrayBuffer) {
    return new TextDecoder().decode(payload);
  }
  return String(payload || "");
}

function updateAgentEouDebug(message, speaker) {
  const probability = Number(message.probability);
  const threshold = Number(message.threshold);
  const endpointingDelay = Number(message.endpointingDelay);
  const result = message.result === "max_delay" ? "max_delay" : "min_delay";
  const mode = result === "max_delay" ? "warn" : "ok";
  setText(el.agentEouStatus, result, mode);
  el.agentEouProbability.textContent = formatValue(probability);
  el.agentEouThreshold.textContent = formatValue(threshold);
  el.agentEouDelay.textContent = Number.isFinite(endpointingDelay)
    ? `${endpointingDelay.toFixed(1)}s`
    : "--";
  el.agentEouResult.textContent = result;
  el.agentEouResult.style.color = result === "max_delay" ? "var(--amber)" : "var(--green)";
  el.agentEouTrigger.textContent = message.trigger || "--";
  el.agentEouSource.textContent = message.endpointMaxSource || "--";
  log(
    `agent EOU ${result} from ${speaker}: p=${formatValue(probability)} threshold=${formatValue(
      threshold
    )} delay=${el.agentEouDelay.textContent} trigger=${message.trigger || "--"} source=${
      message.endpointMaxSource || "--"
    }`
  );
}

function updateLatencyMetrics(message, speaker) {
  const metrics = message.metrics || {};
  const debug = message.debug || {};
  const stt = metricNumber(metrics.audio_tail_to_asr_final_ms ?? metrics.asr);
  const eouWait = metricNumber(metrics.asr_final_to_eou_confirmed_ms ?? metrics.eou_wait);
  const llm = metricNumber(metrics.llm_input_to_first_token_ms ?? metrics.llm);
  const tts = metricNumber(metrics.tts_text_to_first_audio_ms ?? metrics.tts);
  const total = metricNumber(metrics.audio_tail_to_tts_first_audio_ms ?? metrics.total);
  const mode = debug.preemptiveGenerationUsed ? "preemptive" : "regular";

  setText(el.latencyStatus, `seq ${message.seq ?? "--"}`, "ok");
  el.latencyStt.textContent = formatMs(stt);
  el.latencyEouWait.textContent = formatMs(eouWait);
  el.latencyLlm.textContent = formatMs(llm);
  el.latencyTts.textContent = formatMs(tts);
  el.latencyTotal.textContent = formatMs(total);
  el.latencyMode.textContent = mode;
  el.latencyMode.style.color = debug.preemptiveGenerationUsed ? "var(--blue)" : "";
  log(
    `latency seq=${message.seq ?? "--"} from ${speaker}: stt=${formatMs(stt)} eou=${formatMs(
      eouWait
    )} llm=${formatMs(llm)} tts=${formatMs(tts)} total=${formatMs(total)} mode=${mode} anchor=${
      debug.audioTailSource || "--"
    }`
  );
}

function getClientIdentity() {
  if (state.clientIdentity) {
    return state.clientIdentity;
  }
  const storageKey = "livekitAudioTurnDemoIdentity";
  try {
    const existing = window.sessionStorage?.getItem(storageKey);
    if (existing) {
      state.clientIdentity = existing;
      return existing;
    }
  } catch (_error) {
    // Session storage is best-effort; random identity is fine if it is unavailable.
  }
  const random =
    window.crypto?.randomUUID?.().replace(/-/g, "").slice(0, 10) ||
    Math.random().toString(16).slice(2, 12);
  state.clientIdentity = `web-${random}`;
  try {
    window.sessionStorage?.setItem(storageKey, state.clientIdentity);
  } catch (_error) {
    // Ignore storage failures in private browsing modes.
  }
  return state.clientIdentity;
}

function remoteAudioKey(track, publication, participant) {
  return `${participant.identity}:${publication?.trackSid || publication?.sid || track.sid || track.mediaStreamTrack?.id || "audio"}`;
}

function attachRemoteAudio(track, publication, participant) {
  const key = remoteAudioKey(track, publication, participant);
  if (state.remoteAudioElements.has(key)) {
    return;
  }
  const audio = ensureAgentAudioElement();
  let attachedAudio = audio;
  try {
    attachedAudio = track.attach(audio);
  } catch (_error) {
    attachedAudio = track.attach();
    el.remoteAudio.appendChild(attachedAudio);
  }
  attachedAudio.autoplay = true;
  attachedAudio.controls = true;
  attachedAudio.playsInline = true;
  attachedAudio.muted = false;
  attachedAudio.volume = 1;
  audio.autoplay = true;
  audio.controls = true;
  audio.playsInline = true;
  attachedAudio.dataset.participant = participant.identity;
  attachedAudio.dataset.trackSid = publication?.trackSid || publication?.sid || "";
  state.remoteAudioElements.set(key, { audio: attachedAudio, track });
  if (isExpectedAgentParticipant(participant)) {
    state.activeAgentIdentity = participant.identity;
    setAgentState("audio ready", "ok");
  }
  setAudioState("playing", "ok");
  playAgentAudio(attachedAudio);
}

function attachParticipantAudioFromPublications(participant) {
  for (const publication of participant.trackPublications?.values?.() || []) {
    const track = publication.track;
    if (track && isAudioTrack(track)) {
      attachRemoteAudio(track, publication, participant);
    }
  }
}

function detachRemoteAudio(track, publication, participant) {
  const key = remoteAudioKey(track, publication, participant);
  const known = state.remoteAudioElements.get(key);
  for (const audio of track.detach()) {
    if (audio !== state.agentAudioElement) {
      audio.remove();
    }
  }
  if (known) {
    if (known.audio !== state.agentAudioElement) {
      known.audio.remove();
    }
    state.remoteAudioElements.delete(key);
  }
  if (state.remoteAudioElements.size === 0) {
    setAudioState(state.audioUnlocked ? "unlocked" : "locked", state.audioUnlocked ? "ok" : "");
  }
}

function detachParticipantAudio(participant) {
  for (const [key, item] of state.remoteAudioElements.entries()) {
    if (key.startsWith(`${participant.identity}:`)) {
      item.track?.detach?.();
      if (item.audio !== state.agentAudioElement) {
        item.audio.remove();
      }
      state.remoteAudioElements.delete(key);
    }
  }
  if (state.remoteAudioElements.size === 0) {
    setAudioState(state.audioUnlocked ? "unlocked" : "locked", state.audioUnlocked ? "ok" : "");
  }
}

function clearRemoteAudio() {
  for (const item of state.remoteAudioElements.values()) {
    item.track?.detach?.();
    if (item.audio !== state.agentAudioElement) {
      item.audio.remove();
    }
  }
  state.remoteAudioElements.clear();
  if (state.agentAudioElement) {
    state.agentAudioElement.pause();
    state.agentAudioElement.removeAttribute("src");
    state.agentAudioElement.srcObject = null;
    state.agentAudioElement.load();
  }
  setAudioState(state.audioUnlocked ? "unlocked" : "locked", state.audioUnlocked ? "ok" : "");
}

function withTimeout(promise, timeoutMs, message) {
  let timeoutId;
  const timeout = new Promise((_, reject) => {
    timeoutId = window.setTimeout(() => reject(new Error(message)), timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => window.clearTimeout(timeoutId));
}

async function dispatchAgent(roomName, participantIdentity) {
  if (state.agentDispatchRequested) {
    return;
  }
  state.agentDispatchRequested = true;
  state.agentDispatchStartedAt = Date.now();
  state.expectedAgentIdentity = PENDING_AGENT_IDENTITY;
  const params = new URLSearchParams({ room: roomName });
  if (participantIdentity) {
    params.set("participantIdentity", participantIdentity);
  }
  const resp = await fetch(`/dispatch?${params.toString()}`, { method: "POST" });
  if (!resp.ok) {
    state.agentDispatchRequested = false;
    state.expectedAgentIdentity = null;
    throw new Error(`agent dispatch failed: ${await resp.text()}`);
  }
  const payload = await resp.json();
  if (payload.dispatched) {
    setAgentState("dispatching", "active");
    log(
      `agent dispatch requested: ${payload.agentName} dispatch=${payload.id || "--"} for ${
        participantIdentity || "auto"
      }`
    );
    attachExpectedAgentAudioFromRoom();
  } else if (payload.reason !== "disabled") {
    state.expectedAgentIdentity = null;
    setAgentState("dispatch skipped", "warn");
    log(`agent dispatch skipped: ${payload.reason}`);
  } else {
    state.expectedAgentIdentity = null;
  }
}

function attachExpectedAgentAudioFromRoom() {
  if (!state.room || !state.expectedAgentIdentity) {
    return;
  }
  let participant = null;
  if (state.expectedAgentIdentity === PENDING_AGENT_IDENTITY) {
    for (const candidate of state.room.remoteParticipants?.values?.() || []) {
      if (isAgentParticipant(candidate) && acceptExpectedAgentParticipant(candidate)) {
        participant = candidate;
        break;
      }
    }
  } else {
    participant = state.room.remoteParticipants?.get?.(state.expectedAgentIdentity);
  }
  if (!participant) {
    return;
  }
  state.activeAgentIdentity = participant.identity;
  setAgentState("joined", "ok");
  attachParticipantAudioFromPublications(participant);
}

async function startMic() {
  if (!canUseMicrophone()) {
    throw new Error(microphoneUnavailableMessage());
  }
  await unlockAudioPlayback();
  await resumeAgentPlayback();
  beginNewInputRun();
  const track = await createLocalAudioTrack({
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  });
  await state.room.localParticipant.publishTrack(track, {
    name: "mic-audio-turn-demo",
    source: Track.Source.Microphone,
  });
  state.micTrack = track;
  state.activeInput = "mic";
  setInputState("microphone", "active");
  setDialogueState(state.activeAgentIdentity ? "listening" : "waiting agent", "active");
  setControls();
  log("microphone track published");
}

async function startFileReplay(file, options = {}) {
  const inputKind = options.inputKind || "file";
  const inputLabel = options.inputLabel || "file replay";
  const trackPrefix = options.trackPrefix || "file-replay";
  const displayName = options.displayName || file.name || "audio";
  const leadInMs = options.leadInMs ?? REPLAY_LEAD_IN_MS;
  const tailMs = options.tailMs ?? REPLAY_TAIL_MS;
  const autoStopOnEnd = options.autoStopOnEnd ?? true;
  beginNewInputRun();
  await unlockAudioPlayback();
  if (state.fileStopTimer) {
    window.clearTimeout(state.fileStopTimer);
    state.fileStopTimer = null;
  }
  const audio = new Audio();
  audio.src = URL.createObjectURL(file);
  audio.controls = false;
  audio.loop = false;

  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("Web Audio API is unavailable in this browser");
  }
  const audioContext = new AudioContextClass();
  await audioContext.resume?.();
  const source = audioContext.createMediaElementSource(audio);
  const destination = audioContext.createMediaStreamDestination();
  source.connect(destination);
  source.connect(audioContext.destination);
  const silenceSource = createSilenceSource(audioContext, destination);

  const mediaTrack = destination.stream.getAudioTracks()[0];
  await state.room.localParticipant.publishTrack(mediaTrack, {
    name: `${trackPrefix}-${sanitizeTrackName(displayName)}`,
    source: Track.Source.Microphone,
  });
  audio.addEventListener("ended", () => {
    if (!autoStopOnEnd) {
      state.debugTtsPaused = true;
      state.debugTtsEnded = true;
      log(`${inputLabel} ended; holding LiveKit track as silence`);
      setInputState(`${inputLabel} silence`, "active");
      setDialogueState("listening", "active");
      setControls();
      return;
    }
    log(`${inputLabel} ended; holding track for ASR finalization`);
    setInputState(`${inputLabel} finishing`, "active");
    if (state.fileStopTimer) {
      window.clearTimeout(state.fileStopTimer);
    }
    state.fileStopTimer = window.setTimeout(() => {
      state.fileStopTimer = null;
      if (state.fileAudio === audio) {
        stopInput();
      }
    }, tailMs);
  });

  state.fileTrack = mediaTrack;
  state.fileAudio = audio;
  state.fileContext = audioContext;
  state.fileSilenceSource = silenceSource;
  state.activeInput = inputKind;
  state.debugTtsPaused = false;
  state.debugTtsEnded = false;
  setInputState(inputLabel, "active");
  setDialogueState(state.activeAgentIdentity ? "listening" : "waiting agent", "active");
  setControls();
  log(`${inputLabel} track published: ${displayName}`);

  if (leadInMs > 0) {
    await sleep(leadInMs);
  }
  if (state.fileAudio !== audio) {
    return;
  }
  try {
    await audioContext.resume?.();
    await audio.play();
    log(`${inputLabel} playback started: ${displayName}`);
  } catch (error) {
    await stopInput();
    throw error;
  }
}

function createSilenceSource(audioContext, destination) {
  if (typeof audioContext.createConstantSource !== "function") {
    return null;
  }
  const silenceSource = audioContext.createConstantSource();
  const silenceGain = audioContext.createGain();
  silenceGain.gain.value = 0;
  silenceSource.connect(silenceGain);
  silenceGain.connect(destination);
  silenceSource.start();
  return silenceSource;
}

async function stopInput() {
  if (!state.room) {
    return;
  }
  if (state.micTrack) {
    await state.room.localParticipant.unpublishTrack(state.micTrack, true);
    state.micTrack = null;
  }
  if (state.fileTrack) {
    await state.room.localParticipant.unpublishTrack(state.fileTrack, true);
    state.fileTrack = null;
  }
  if (state.debugSimTrack) {
    await state.room.localParticipant.unpublishTrack(state.debugSimTrack, true);
    state.debugSimTrack = null;
  }
  if (state.fileStopTimer) {
    window.clearTimeout(state.fileStopTimer);
    state.fileStopTimer = null;
  }
  if (state.fileAudio) {
    state.fileAudio.pause();
    URL.revokeObjectURL(state.fileAudio.src);
    state.fileAudio = null;
  }
  if (state.fileSilenceSource) {
    try {
      state.fileSilenceSource.stop();
    } catch (_error) {
      // The source may already be stopped when cleaning up after an ended simulation.
    }
    state.fileSilenceSource = null;
  }
  if (state.fileContext) {
    await state.fileContext.close();
    state.fileContext = null;
  }
  clearDebugTtsClip();
  if (state.debugSimSilenceSource) {
    try {
      state.debugSimSilenceSource.stop();
    } catch (_error) {
      // The source may already be stopped while cleaning up.
    }
    state.debugSimSilenceSource = null;
  }
  if (state.debugSimContext) {
    await state.debugSimContext.close();
    state.debugSimContext = null;
  }
  state.debugSimDestination = null;
  state.debugTtsPaused = false;
  state.debugTtsEnded = false;
  state.debugTtsPlaying = false;
  state.activeInput = null;
  state.stoppedSinceLastRun = true;
  state.acceptingPoints = false;
  setInputState("idle");
  setDialogueState(state.connected ? "ready" : "offline", state.connected ? "ok" : "");
  setControls();
  log("input stopped");
}

async function synthesizeDebugTts() {
  const text = el.debugTtsText.value.trim();
  if (!text) {
    throw new Error("debug TTS text is empty");
  }
  await unlockAudioPlayback();
  state.debugTtsSynthesizing = true;
  setInputState("synthesizing", "active");
  setControls();
  log("debug TTS synthesis requested");
  try {
    const response = await fetch("/debug/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const blob = await response.blob();
    if (!blob.size) {
      throw new Error("debug TTS returned empty audio");
    }
    state.debugTtsBlob = blob;
    state.debugTtsPreparedText = text;
    state.debugTtsEnded = false;
    state.debugTtsSynthesizing = false;
    restoreDebugTtsInputState();
    setControls();
    log(`debug TTS audio ready: ${text.slice(0, 48)}`);
  } catch (error) {
    state.debugTtsSynthesizing = false;
    restoreDebugTtsInputState();
    setControls();
    throw error;
  }
}

async function startDebugTtsSimulation() {
  if (state.activeInput === "debug-tts") {
    return;
  }
  if (state.activeInput) {
    throw new Error("another input is already active");
  }
  beginNewInputRun();
  await unlockAudioPlayback();
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("Web Audio API is unavailable in this browser");
  }
  const audioContext = new AudioContextClass();
  await audioContext.resume?.();
  const destination = audioContext.createMediaStreamDestination();
  const silenceSource = createSilenceSource(audioContext, destination);
  const mediaTrack = destination.stream.getAudioTracks()[0];
  await state.room.localParticipant.publishTrack(mediaTrack, {
    name: `debug-tts-simulation-${Date.now()}`,
    source: Track.Source.Microphone,
  });

  state.debugSimTrack = mediaTrack;
  state.debugSimContext = audioContext;
  state.debugSimDestination = destination;
  state.debugSimSilenceSource = silenceSource;
  state.activeInput = "debug-tts";
  state.debugTtsPaused = false;
  state.debugTtsEnded = false;
  state.debugTtsPlaying = false;
  setInputState("debug TTS silence", "active");
  setDialogueState(state.activeAgentIdentity ? "listening" : "waiting agent", "active");
  setControls();
  log("debug TTS simulation track published; holding silence");
}

async function startDebugTtsReplay() {
  if (state.activeInput === "debug-tts" && state.debugTtsPaused) {
    await resumeDebugTtsSimulation();
    return;
  }
  if (!hasPreparedDebugTts()) {
    throw new Error("debug TTS audio is not ready; synthesize first");
  }
  if (state.activeInput && state.activeInput !== "debug-tts") {
    throw new Error("another input is already active");
  }
  if (state.activeInput !== "debug-tts") {
    await startDebugTtsSimulation();
  }
  if (!state.debugSimContext || !state.debugSimDestination) {
    throw new Error("debug TTS simulation is not ready");
  }
  clearDebugTtsClip();
  beginNewInputRun({ keepTrack: true });
  const text = state.debugTtsPreparedText;
  const audioUrl = URL.createObjectURL(state.debugTtsBlob);
  const audio = new Audio();
  audio.src = audioUrl;
  audio.controls = false;
  audio.loop = false;
  const source = state.debugSimContext.createMediaElementSource(audio);
  source.connect(state.debugSimDestination);
  source.connect(state.debugSimContext.destination);
  audio.addEventListener("ended", handleDebugTtsClipEnded, { once: true });

  state.debugSimAudio = audio;
  state.debugSimAudioSource = source;
  state.debugSimAudioUrl = audioUrl;
  state.debugTtsPaused = false;
  state.debugTtsEnded = false;
  state.debugTtsPlaying = true;
  setInputState("debug TTS replay", "active");
  setDialogueState(state.activeAgentIdentity ? "listening" : "waiting agent", "active");
  setControls();
  await state.debugSimContext.resume?.();
  await audio.play();
  log(`debug TTS replay started: ${text.slice(0, 48)}`);
}

async function resumeDebugTtsSimulation() {
  if (state.activeInput !== "debug-tts" || !state.debugSimAudio || !state.debugTtsPaused) {
    return;
  }
  await state.debugSimContext?.resume?.();
  await state.debugSimAudio.play();
  state.debugTtsPaused = false;
  state.debugTtsPlaying = true;
  setInputState("debug TTS simulation", "active");
  setDialogueState(state.activeAgentIdentity ? "listening" : "waiting agent", "active");
  setControls();
  log("debug TTS simulation resumed");
}

function pauseDebugTtsAsSilence() {
  if (
    state.activeInput !== "debug-tts" ||
    !state.debugSimAudio ||
    state.debugTtsPaused ||
    !state.debugTtsPlaying
  ) {
    return;
  }
  state.debugSimAudio.pause();
  state.debugTtsPaused = true;
  setInputState("debug TTS silence", "active");
  setDialogueState("listening", "active");
  setControls();
  log("debug TTS simulation paused; LiveKit track is still publishing silence");
}

async function endDebugTtsSimulation() {
  if (state.activeInput !== "debug-tts") {
    return;
  }
  await stopInput();
  log("debug TTS simulation ended");
}

function handleDebugTtsClipEnded() {
  clearDebugTtsClip({ keepEndedState: true });
  setInputState("debug TTS silence", "active");
  setDialogueState("listening", "active");
  setControls();
  log("debug TTS replay ended; simulation track is holding silence");
}

function clearDebugTtsClip(options = {}) {
  const keepEndedState = Boolean(options.keepEndedState);
  if (state.debugSimAudio) {
    state.debugSimAudio.pause();
    state.debugSimAudio.removeEventListener("ended", handleDebugTtsClipEnded);
    state.debugSimAudio.removeAttribute("src");
    state.debugSimAudio.load?.();
    state.debugSimAudio = null;
  }
  if (state.debugSimAudioSource) {
    withSuppressedErrors(() => state.debugSimAudioSource.disconnect());
    state.debugSimAudioSource = null;
  }
  if (state.debugSimAudioUrl) {
    URL.revokeObjectURL(state.debugSimAudioUrl);
    state.debugSimAudioUrl = "";
  }
  state.debugTtsPaused = false;
  state.debugTtsPlaying = false;
  state.debugTtsEnded = keepEndedState;
}

function sanitizeTrackName(value) {
  return String(value || "audio")
    .replace(/[^A-Za-z0-9_.-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64) || "audio";
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function withSuppressedErrors(action) {
  try {
    action();
  } catch (_error) {
    // Cleanup helpers should be best-effort.
  }
}

function beginNewInputRun(options = {}) {
  const keepTrack = Boolean(options.keepTrack);
  resetSeries({ keepTrack });
  state.stoppedSinceLastRun = false;
  if (!keepTrack) {
    state.activeTrackSid = null;
  }
  state.acceptingPoints = true;
}

function resetSeries(options = {}) {
  const keepTrack = Boolean(options.keepTrack);
  const activeTrackSid = state.activeTrackSid;
  state.points = [];
  state.threshold = null;
  state.pointTimeOffset = null;
  state.activeTrackSid = keepTrack ? activeTrackSid : null;
  el.score.textContent = "--";
  el.threshold.textContent = "--";
  el.decision.textContent = "waiting";
  el.decision.style.color = "var(--muted)";
  el.time.textContent = "0.0s";
  drawEmptyTimeline();
}

function connectEvents() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${window.location.host}/events`);
  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    handleEvent(payload);
  };
  ws.onclose = () => {
    setTimeout(connectEvents, 1000);
  };
}

function handleEvent(payload) {
  if (payload.type === "status") {
    log(payload.message);
    if (payload.level === "error") {
      setBadge("error", "error");
    }
    return;
  }
  if (payload.type === "track_started") {
    resetSeries();
    state.activeTrackSid = payload.trackSid;
    state.acceptingPoints = true;
    state.threshold = payload.threshold;
    el.threshold.textContent = formatValue(payload.threshold);
    log(`detector started for ${payload.participant}`);
    if (state.activeInput) {
      setDialogueState("listening", "active");
    }
    return;
  }
  if (payload.type === "track_ended") {
    if (payload.trackSid === state.activeTrackSid) {
      state.acceptingPoints = false;
    }
    log(`detector track ended at ${payload.time}s`);
    if (state.activeInput) {
      setDialogueState("waiting turn", "active");
    }
    return;
  }
  if (payload.type !== "point") {
    return;
  }
  if (!state.acceptingPoints || payload.trackSid !== state.activeTrackSid) {
    return;
  }

  state.threshold = payload.threshold;
  const point = pointWithDisplayTime(payload);
  addPoint(point);
  if (state.points.length > 2500) {
    state.points.shift();
  }
  el.score.textContent = formatValue(payload.score);
  el.threshold.textContent = formatValue(payload.threshold);
  el.decision.textContent = payload.decision ? "end" : "continue";
  el.decision.style.color = payload.decision ? "var(--green)" : "var(--muted)";
  if (payload.decision && state.activeInput) {
    setDialogueState("turn detected", "ok");
  }
  el.time.textContent = `${point.time.toFixed(1)}s`;
  draw();
}

function pointWithDisplayTime(payload) {
  if (state.pointTimeOffset === null) {
    state.pointTimeOffset = payload.time;
  }
  return {
    ...payload,
    rawTime: payload.time,
    time: Math.max(0, payload.time - state.pointTimeOffset),
  };
}

function resetConversationView() {
  clearLiveBubbleTimer("user");
  clearLiveBubbleTimer("assistant");
  clearAssistantReadyTimer();
  resetAgentEouDebug();
  resetLatencyMetrics();
  el.userTranscript.textContent = "--";
  el.userTranscript.classList.add("empty");
  el.userTranscriptStatus.textContent = "waiting";
  el.assistantTranscript.textContent = "--";
  el.assistantTranscript.classList.add("empty");
  el.assistantTranscriptStatus.textContent = "waiting";
  setText(el.conversationStatus, "ready", "ok");
  el.conversationFeed.textContent = "";
  state.transcriptSeen.clear();
  state.transcriptContentSeen.clear();
  state.liveBubbles.user = null;
  state.liveBubbles.assistant = null;
}

function resetAgentEouDebug() {
  setText(el.agentEouStatus, "waiting");
  el.agentEouProbability.textContent = "--";
  el.agentEouThreshold.textContent = "--";
  el.agentEouDelay.textContent = "--";
  el.agentEouResult.textContent = "waiting";
  el.agentEouResult.style.color = "";
  el.agentEouTrigger.textContent = "--";
  el.agentEouSource.textContent = "--";
}

function resetLatencyMetrics() {
  setText(el.latencyStatus, "waiting");
  el.latencyStt.textContent = "--";
  el.latencyEouWait.textContent = "--";
  el.latencyLlm.textContent = "--";
  el.latencyTts.textContent = "--";
  el.latencyTotal.textContent = "--";
  el.latencyMode.textContent = "--";
  el.latencyMode.style.color = "";
}

function handleTranscript(speaker, text, isFinal, segmentId = "") {
  const role = speaker.startsWith("agent-") ? "assistant" : "user";
  if (!shouldAcceptTranscript(role, speaker, text, isFinal, segmentId)) {
    return;
  }
  if (role === "user") {
    el.userTranscript.textContent = text;
    el.userTranscript.classList.remove("empty");
    el.userTranscriptStatus.textContent = isFinal ? "final" : "streaming";
    setDialogueState(isFinal ? "thinking" : "user speaking", isFinal ? "active" : "warn");
  } else {
    el.assistantTranscript.textContent = text;
    el.assistantTranscript.classList.remove("empty");
    el.assistantTranscriptStatus.textContent = isFinal ? "final" : "streaming";
    clearAssistantReadyTimer();
    setDialogueState("assistant speaking", "active");
    setAgentState("speaking", "active");
    if (isFinal) {
      scheduleAssistantReadyState();
    }
  }
  updateConversationBubble(role, text, isFinal, segmentId);
}

function shouldAcceptTranscript(role, speaker, text, isFinal, segmentId = "") {
  const normalized = normalizeTranscript(text);
  if (!normalized) {
    return false;
  }
  if (isStaleTranscriptPrefix(role, normalized)) {
    return false;
  }
  if (role === "assistant") {
    if (
      state.expectedAgentIdentity === PENDING_AGENT_IDENTITY &&
      state.agentDispatchRequested &&
      !state.knownAgentIdentities.has(speaker)
    ) {
      acceptExpectedAgentIdentity(speaker);
    }
    if (
      state.expectedAgentIdentity === PENDING_AGENT_IDENTITY ||
      (state.expectedAgentIdentity && speaker !== state.expectedAgentIdentity)
    ) {
      log(`ignored stale agent transcript ${speaker}: ${normalized}`);
      return false;
    }
  } else if (state.clientIdentity && speaker !== state.clientIdentity && speaker !== "unknown") {
    log(`ignored non-local user transcript ${speaker}: ${normalized}`);
    return false;
  }

  const now = Date.now();
  pruneTranscriptSeen(now);
  const kind = isFinal ? "final" : "interim";
  const segmentKey = `${role}:${segmentId || "no-segment"}:${kind}:${normalized}`;
  if (state.transcriptSeen.has(segmentKey)) {
    return false;
  }
  state.transcriptSeen.set(segmentKey, now);

  const contentKey = `${role}:${kind}:${normalized}`;
  const previous = state.transcriptContentSeen.get(contentKey);
  const windowMs = isFinal ? TRANSCRIPT_DEDUPE_MS : 1200;
  if (previous && now - previous < windowMs) {
    return false;
  }
  state.transcriptContentSeen.set(contentKey, now);
  return true;
}

function normalizeTranscript(text) {
  return String(text || "").replace(/\s+/g, " ").trim();
}

function compactTranscript(text) {
  return normalizeTranscript(text).replace(/\s+/g, "");
}

function getBubbleText(bubble) {
  return bubble?.querySelector(".message-text")?.textContent || "";
}

function isStaleTranscriptPrefix(role, normalizedText) {
  const current = compactTranscript(getBubbleText(state.liveBubbles[role]));
  const incoming = compactTranscript(normalizedText);
  return Boolean(current && incoming && current.length > incoming.length && current.startsWith(incoming));
}

function pruneTranscriptSeen(now) {
  for (const [key, timestamp] of state.transcriptSeen.entries()) {
    if (now - timestamp > TRANSCRIPT_DEDUPE_MS) {
      state.transcriptSeen.delete(key);
    }
  }
  for (const [key, timestamp] of state.transcriptContentSeen.entries()) {
    if (now - timestamp > TRANSCRIPT_DEDUPE_MS) {
      state.transcriptContentSeen.delete(key);
    }
  }
}

function updateConversationBubble(role, text, isFinal, segmentId = "") {
  clearLiveBubbleTimer(role);
  let bubble = state.liveBubbles[role];
  if (!canReuseConversationBubble(bubble, text, segmentId)) {
    bubble = document.createElement("div");
    bubble.className = `message ${role}`;
    bubble.dataset.final = "false";
    bubble.dataset.segmentId = segmentId;

    const roleLabel = document.createElement("span");
    roleLabel.className = "message-role";
    roleLabel.textContent = role === "assistant" ? "Assistant" : "User";
    const body = document.createElement("div");
    body.className = "message-text";
    bubble.append(roleLabel, body);
    el.conversationFeed.appendChild(bubble);
    state.liveBubbles[role] = bubble;
  }

  bubble.querySelector(".message-text").textContent = mergedTranscriptText(getBubbleText(bubble), text);
  bubble.classList.toggle("interim", !isFinal);
  bubble.dataset.segmentId = segmentId || bubble.dataset.segmentId || "";
  bubble.dataset.final = isFinal ? "true" : "false";
  setText(el.conversationStatus, isFinal ? "final" : "streaming", isFinal ? "ok" : "active");
  if (isFinal) {
    releaseLiveBubbleAfterFinal(role, bubble);
  }
  el.conversationFeed.scrollTop = el.conversationFeed.scrollHeight;
}

function canReuseConversationBubble(bubble, text, segmentId = "") {
  if (!bubble) {
    return false;
  }
  const currentText = getBubbleText(bubble);
  if (transcriptsLookRelated(currentText, text)) {
    return true;
  }
  return Boolean(segmentId && bubble.dataset.segmentId === segmentId);
}

function mergedTranscriptText(currentText, incomingText) {
  const current = compactTranscript(currentText);
  const incoming = compactTranscript(incomingText);
  if (current && incoming && current.startsWith(incoming)) {
    return currentText;
  }
  return incomingText;
}

function transcriptsLookRelated(previousText, nextText) {
  const previous = compactTranscript(previousText);
  const next = compactTranscript(nextText);
  if (!previous || !next) {
    return false;
  }
  if (previous === next || previous.startsWith(next) || next.startsWith(previous)) {
    return true;
  }
  const minLength = Math.min(previous.length, next.length);
  if (minLength < 8) {
    return false;
  }
  let commonPrefix = 0;
  while (commonPrefix < minLength && previous[commonPrefix] === next[commonPrefix]) {
    commonPrefix += 1;
  }
  return commonPrefix / minLength >= 0.75;
}

function releaseLiveBubbleAfterFinal(role, bubble) {
  clearLiveBubbleTimer(role);
  state.liveBubbleTimers[role] = window.setTimeout(() => {
    if (state.liveBubbles[role] === bubble) {
      state.liveBubbles[role] = null;
    }
    state.liveBubbleTimers[role] = null;
    if (!state.liveBubbles.user && !state.liveBubbles.assistant) {
      setText(el.conversationStatus, "ready", "ok");
    }
  }, FINAL_BUBBLE_SETTLE_MS);
}

function clearLiveBubbleTimer(role) {
  if (!state.liveBubbleTimers[role]) {
    return;
  }
  window.clearTimeout(state.liveBubbleTimers[role]);
  state.liveBubbleTimers[role] = null;
}

function scheduleAssistantReadyState() {
  clearAssistantReadyTimer();
  state.assistantReadyTimer = window.setTimeout(() => {
    state.assistantReadyTimer = null;
    if (!state.connected || !state.activeAgentIdentity) {
      return;
    }
    setAgentState("ready", "ok");
    if (!state.activeInput) {
      setDialogueState("ready", "ok");
    }
    if (state.remoteAudioElements.size > 0) {
      setAudioState("ready", "ok");
    }
  }, ASSISTANT_READY_SETTLE_MS);
}

function clearAssistantReadyTimer() {
  if (!state.assistantReadyTimer) {
    return;
  }
  window.clearTimeout(state.assistantReadyTimer);
  state.assistantReadyTimer = null;
}

function ensureAgentAudioElement() {
  if (state.agentAudioElement) {
    if (!state.agentAudioElement.isConnected) {
      el.remoteAudio.appendChild(state.agentAudioElement);
    }
    return state.agentAudioElement;
  }
  const audio = document.createElement("audio");
  audio.id = "agentAudioPlayer";
  audio.autoplay = true;
  audio.controls = true;
  audio.playsInline = true;
  audio.preload = "auto";
  el.remoteAudio.appendChild(audio);
  state.agentAudioElement = audio;
  return audio;
}

async function unlockAudioPlayback() {
  ensureAgentAudioElement();
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (AudioContextClass && !state.audioContext) {
    state.audioContext = new AudioContextClass();
  }
  try {
    await state.audioContext?.resume?.();
  } catch (error) {
    log(`audio context unlock failed: ${error.message}`);
  }

  if (state.audioUnlocked) {
    setAudioState("unlocked", "ok");
    return;
  }

  const audio = state.agentAudioElement;
  const silentUrl = createSilentWavUrl();
  try {
    audio.muted = true;
    audio.srcObject = null;
    audio.src = silentUrl;
    await audio.play();
    audio.pause();
    audio.removeAttribute("src");
    audio.load();
    audio.muted = false;
    state.audioUnlocked = true;
    setAudioState("unlocked", "ok");
  } catch (error) {
    audio.muted = false;
    setAudioState("click required", "warn");
    log(`audio unlock pending: ${error.message}`);
  } finally {
    URL.revokeObjectURL(silentUrl);
  }
}

function createSilentWavUrl() {
  const sampleRate = 8000;
  const samples = 80;
  const buffer = new ArrayBuffer(44 + samples * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples * 2, true);
  return URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
}

function writeAscii(view, offset, value) {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

async function startRoomAudio(room) {
  if (typeof room.startAudio === "function") {
    try {
      await room.startAudio();
      setAudioState(state.audioUnlocked ? "unlocked" : "ready", "ok");
    } catch (error) {
      setAudioState("click required", "warn");
      log(`room audio start pending: ${error.message}`);
    }
  }
}

async function playAgentAudio(audio) {
  try {
    await startRoomAudio(state.room);
    await audio.play();
    setAudioState("playing", "ok");
  } catch (error) {
    setAudioState("click required", "warn");
    log(`agent audio autoplay blocked: ${error.message}`);
  }
}

async function resumeAgentPlayback() {
  if (!state.agentAudioElement) {
    return;
  }
  await playAgentAudio(state.agentAudioElement);
}

function addPoint(point) {
  const index = state.points.findIndex((p) => p.time > point.time);
  if (index === -1) {
    state.points.push(point);
  } else {
    state.points.splice(index, 0, point);
  }
}

function formatValue(value) {
  return typeof value === "number" ? value.toFixed(3) : "--";
}

function metricNumber(value) {
  if (value === null || value === undefined || value === "") {
    return NaN;
  }
  return Number(value);
}

function formatMs(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(2)}s`;
  }
  return `${Math.round(value)}ms`;
}

function draw() {
  const width = el.canvas.width;
  const height = el.canvas.height;
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 54, right: 22, top: 24, bottom: 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const points = state.points;
  const maxTime = Math.max(8, points.at(-1)?.time ?? 8);
  const minTime = Math.max(0, maxTime - 30);
  const visible = points.filter((p) => p.time >= minTime);

  drawGrid(pad, plotW, plotH, minTime, maxTime);
  drawAudio(visible, pad, plotW, plotH, minTime, maxTime);
  drawThreshold(pad, plotW, plotH);
  drawScore(visible, pad, plotW, plotH, minTime, maxTime);
  drawDecision(visible, pad, plotW, plotH, minTime, maxTime);
}

function drawEmptyTimeline() {
  const width = el.canvas.width;
  const height = el.canvas.height;
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 54, right: 22, top: 24, bottom: 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  drawGrid(pad, plotW, plotH, 0, 8);
}

function xFor(t, pad, plotW, minTime, maxTime) {
  return pad.left + ((t - minTime) / (maxTime - minTime)) * plotW;
}

function yFor(v, pad, plotH) {
  return pad.top + (1 - Math.max(0, Math.min(1, v))) * plotH;
}

function drawGrid(pad, plotW, plotH, minTime, maxTime) {
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  ctx.font = "12px Inter, sans-serif";
  ctx.fillStyle = "#647084";

  for (let i = 0; i <= 5; i += 1) {
    const y = pad.top + (i / 5) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.fillText((1 - i / 5).toFixed(1), 12, y + 4);
  }

  const start = Math.ceil(minTime / 5) * 5;
  for (let t = start; t <= maxTime; t += 5) {
    const x = xFor(t, pad, plotW, minTime, maxTime);
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();
    ctx.fillText(`${t.toFixed(0)}s`, x - 10, pad.top + plotH + 24);
  }
}

function drawAudio(points, pad, plotW, plotH, minTime, maxTime) {
  if (points.length === 0) {
    return;
  }
  ctx.fillStyle = "rgba(148, 163, 184, 0.35)";
  for (const p of points) {
    const x = xFor(p.time, pad, plotW, minTime, maxTime);
    const h = Math.max(2, p.rms * plotH * 1.8);
    ctx.fillRect(x, pad.top + plotH - h, 3, h);
  }
}

function drawThreshold(pad, plotW, plotH) {
  if (typeof state.threshold !== "number") {
    return;
  }
  const y = yFor(state.threshold, pad, plotH);
  ctx.strokeStyle = "#b7791f";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.moveTo(pad.left, y);
  ctx.lineTo(pad.left + plotW, y);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawScore(points, pad, plotW, plotH, minTime, maxTime) {
  if (points.length < 2) {
    return;
  }
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  points.forEach((p, index) => {
    const x = xFor(p.time, pad, plotW, minTime, maxTime);
    const y = yFor(p.score, pad, plotH);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function drawDecision(points, pad, plotW, plotH, minTime, maxTime) {
  if (points.length < 2) {
    return;
  }
  const high = 0.95;
  const low = 0.08;
  ctx.strokeStyle = "#11845b";
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, index) => {
    const x = xFor(p.time, pad, plotW, minTime, maxTime);
    const y = yFor(p.decision ? high : low, pad, plotH);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      const prev = points[index - 1];
      const prevY = yFor(prev.decision ? high : low, pad, plotH);
      ctx.lineTo(x, prevY);
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

el.connectBtn.addEventListener("click", async () => {
  try {
    await unlockAudioPlayback();
    await connectRoom();
  } catch (error) {
    state.connecting = false;
    state.connected = false;
    setControls();
    setBadge("error", "error");
    log(`connect failed: ${error.message}`);
  }
});

el.micBtn.addEventListener("click", async () => {
  try {
    await unlockAudioPlayback();
    await startMic();
  } catch (error) {
    log(`microphone failed: ${error.message}`);
  }
});

el.fileInput.addEventListener("change", async () => {
  const file = el.fileInput.files?.[0];
  el.fileInput.value = "";
  if (!file) {
    return;
  }
  try {
    await startFileReplay(file);
  } catch (error) {
    log(`file replay failed: ${error.message}`);
  }
});

el.debugTtsText.addEventListener("input", setControls);

el.debugTtsSessionBtn.addEventListener("click", async () => {
  try {
    await startDebugTtsSimulation();
  } catch (error) {
    log(`debug TTS simulation start failed: ${error.message}`);
  }
});

el.debugTtsBtn.addEventListener("click", async () => {
  try {
    await synthesizeDebugTts();
  } catch (error) {
    log(`debug TTS synthesis failed: ${error.message}`);
  }
});

el.debugTtsStartBtn.addEventListener("click", async () => {
  try {
    await startDebugTtsReplay();
  } catch (error) {
    log(`debug TTS replay start failed: ${error.message}`);
  }
});

el.debugTtsPauseBtn.addEventListener("click", () => {
  try {
    pauseDebugTtsAsSilence();
  } catch (error) {
    log(`debug TTS simulation pause failed: ${error.message}`);
  }
});

el.debugTtsEndBtn.addEventListener("click", async () => {
  try {
    await endDebugTtsSimulation();
  } catch (error) {
    log(`debug TTS simulation end failed: ${error.message}`);
  }
});

el.stopBtn.addEventListener("click", () => {
  stopInput();
});

window.addEventListener("resize", draw);
connectEvents();
if (!canUseMicrophone()) {
  log(microphoneUnavailableMessage());
}
setControls();
drawEmptyTimeline();
