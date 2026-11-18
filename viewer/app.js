(function () {
  "use strict";

  const data = window.RUN_DATA;
  const audio = document.getElementById("audio");
  const timeline = document.getElementById("timeline");
  const playhead = document.getElementById("playhead");
  const duration = Math.max(data.duration_ms, 1);

  function percent(timeMs) {
    return `${Math.min(100, Math.max(0, (timeMs / duration) * 100))}%`;
  }

  function clock(timeMs) {
    const seconds = Math.floor(timeMs / 1000);
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}.${String(
      Math.floor((timeMs % 1000) / 100),
    )}`;
  }

  function seek(timeMs) {
    const target = timeMs / 1000;
    const apply = function () {
      audio.currentTime = target;
      updatePlayhead();
    };
    if (audio.readyState >= 1) {
      apply();
    } else {
      audio.addEventListener("loadedmetadata", apply, { once: true });
    }
  }

  function updatePlayhead() {
    playhead.style.left = percent(audio.currentTime * 1000);
  }

  function segment(item, role) {
    const button = document.createElement("button");
    const label = `${role} speech at ${clock(item.start_ms)}`;
    button.className = `segment ${role}${item.interrupted ? " interrupted" : ""}`;
    button.type = "button";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.style.left = percent(item.start_ms);
    button.style.width = `${(Math.max(item.end_ms - item.start_ms, 0) / duration) * 100}%`;
    button.dataset.startMs = String(item.start_ms);
    button.addEventListener("click", function (event) {
      event.stopPropagation();
      seek(item.start_ms);
    });
    timeline.append(button);
  }

  function marker(item) {
    const button = document.createElement("button");
    const label = `${item.label} at ${clock(item.time_ms)}`;
    button.className = "marker";
    button.type = "button";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.style.left = percent(item.time_ms);
    button.dataset.timeMs = String(item.time_ms);
    button.addEventListener("click", function (event) {
      event.stopPropagation();
      seek(item.time_ms);
    });
    timeline.append(button);
  }

  function row(listId, timeMs, text) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    const time = document.createElement("time");
    const body = document.createElement("span");
    button.type = "button";
    button.dataset.timeMs = String(timeMs);
    time.textContent = clock(timeMs);
    body.textContent = text;
    button.append(time, body);
    button.addEventListener("click", function () {
      seek(timeMs);
    });
    item.append(button);
    document.getElementById(listId).append(item);
  }

  document.getElementById("title").textContent = `${data.scenario_id} · ${data.model_name}`;
  document.getElementById("summary").textContent =
    `${data.suite} · ${data.condition} · ${data.status} · ${clock(data.duration_ms)}`;

  const select = document.getElementById("track");
  data.tracks.forEach(function (track) {
    const option = document.createElement("option");
    option.value = track.file;
    option.textContent = track.label;
    select.append(option);
  });
  select.addEventListener("change", function () {
    const time = audio.currentTime;
    audio.src = select.value;
    audio.addEventListener(
      "loadedmetadata",
      function () {
        audio.currentTime = time;
      },
      { once: true },
    );
  });
  audio.src = select.value;
  audio.addEventListener("timeupdate", updatePlayhead);
  audio.addEventListener("seeked", updatePlayhead);

  data.counterpart.forEach(function (item) {
    segment(item, "counterpart");
  });
  data.model.forEach(function (item) {
    segment(item, "model");
  });
  data.events.forEach(function (item) {
    marker(item);
    row("events", item.time_ms, item.label);
  });
  data.transcript.forEach(function (item) {
    row("transcript", item.time_ms, `${item.speaker}: ${item.text}`);
  });

  timeline.addEventListener("click", function (event) {
    const box = timeline.getBoundingClientRect();
    seek(((event.clientX - box.left) / box.width) * duration);
  });
})();
