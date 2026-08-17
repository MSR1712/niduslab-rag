const uploadBtn = document.getElementById("upload-btn");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const controls = document.getElementById("controls");
const chat = document.getElementById("chat");
const messages = document.getElementById("messages");
const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question-input");
const searchModeSelect = document.getElementById("search-mode");
const streamToggle = document.getElementById("stream-toggle");

let docId = null;
const sessionId = crypto.randomUUID();

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  uploadStatus.textContent = "Uploading & indexing...";
  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || "Upload failed");
    const data = await res.json();
    docId = data.doc_id;
    uploadStatus.textContent = `Indexed "${data.filename}" — ${data.num_pages} page(s), ${data.num_chunks} chunk(s).`;
    controls.hidden = false;
    chat.hidden = false;
  } catch (err) {
    uploadStatus.textContent = `Error: ${err.message}`;
  }
});

function addMessage(role, html) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = html;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function renderSources(sources) {
  if (!sources || sources.length === 0) return "";
  const items = sources
    .map(
      (s) => `<div class="source-item">
        <div class="meta">Page ${s.page} · ${s.matched_by.join("+") || "match"}</div>
        <div>"${s.snippet.replace(/</g, "&lt;")}"</div>
      </div>`
    )
    .join("");
  return `<div class="sources"><strong>Sources</strong>${items}</div>`;
}

askForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question || !docId) return;
  questionInput.value = "";
  addMessage("user", question);

  const payload = {
    doc_id: docId,
    question,
    session_id: sessionId,
    search_mode: searchModeSelect.value,
  };

  if (streamToggle.checked) {
    await askStreaming(payload);
  } else {
    await askNonStreaming(payload);
  }
});

async function askNonStreaming(payload) {
  const answerDiv = addMessage("answer", "Thinking...");
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    answerDiv.innerHTML = `${data.answer}${renderSources(data.sources)}`;
  } catch (err) {
    answerDiv.textContent = `Error: ${err.message}`;
  }
}

async function askStreaming(payload) {
  const answerDiv = addMessage("answer", "");
  let answerText = "";
  let sourcesHtml = "";

  try {
    const res = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const events = buffer.split("\n\n");
      buffer = events.pop(); // keep the last, possibly-incomplete event

      for (const raw of events) {
        const eventMatch = raw.match(/^event: (\w+)/m);
        const dataMatch = raw.match(/^data: (.*)$/m);
        if (!eventMatch || !dataMatch) continue;
        const eventType = eventMatch[1];
        const data = JSON.parse(dataMatch[1]);

        if (eventType === "sources") {
          sourcesHtml = renderSources(
            data.map((s) => ({ ...s, matched_by: s.matched_by || [] }))
          );
        } else if (eventType === "token") {
          answerText += data.text;
          answerDiv.innerHTML = answerText + sourcesHtml;
          messages.scrollTop = messages.scrollHeight;
        }
      }
    }
  } catch (err) {
    answerDiv.textContent = `Error: ${err.message}`;
  }
}
