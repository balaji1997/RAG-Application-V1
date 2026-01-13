const questionInput = document.getElementById("question");
const fileInput = document.getElementById("file");

// ------------------------
// Helper: Add chat bubble
// ------------------------
function add(role, text) {
  const chat = document.getElementById("chat");
  const div = document.createElement("div");

  div.classList.add("chat-bubble", role);
  div.innerText = text;

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;

  return div; // return node for later update
}

// ------------------------
// INDEX DOCUMENT
// ------------------------
async function indexDoc() {
  const file = fileInput.files[0];
  if (!file) {
    alert("Please choose a file first");
    return;
  }

  const form = new FormData();
  form.append("file", file);

  add("bot", "Uploading document...");

  await fetch("/index", {
    method: "POST",
    body: form,
  });

  add("bot", "Document uploaded successfully");
}

// ------------------------
// ASK QUESTION
// ------------------------
async function ask() {
  const q = questionInput.value.trim();
  if (!q) return;

  add("user", q);
  questionInput.value = "";

  // Show thinking indicator
  const thinkingBubble = document.createElement("div");
  thinkingBubble.className = "chat-bubble bot";
  thinkingBubble.innerText = "Thinking...";
  document.getElementById("chat").appendChild(thinkingBubble);

  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: q }),
  });

  const data = await res.json();

  // Remove thinking
  thinkingBubble.remove();

  // 1. Answer
  add("bot", data.answer);

  // 2. Evaluation
  if (data.evaluation) {
    add(
      "evaluation",
      `Evaluation
Faithfulness: ${data.evaluation.faithfulness_score}
Hallucination: ${data.evaluation.hallucination}

${data.evaluation.explanation}`
    );
  }

  // 3. Sources
  if (data.sources && data.sources.length > 0) {
    let sourceText = "Sources\n";

    data.sources.forEach((src, idx) => {
      sourceText += `\n${idx + 1}. ${src.source}`;
      if (src.page !== null && src.page !== undefined) {
        sourceText += ` (page ${src.page})`;
      }
      sourceText += `\n"${src.snippet}"\n`;
    });

    add("sources", sourceText);
  }
}


// ENTER key support
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask();
});
