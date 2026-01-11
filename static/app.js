const questionInput = document.getElementById("question");
const fileInput = document.getElementById("file");

function add(role, text) {
  const chat = document.getElementById("chat");
  const div = document.createElement("div");

  div.classList.add("chat-bubble");

  if (role === "user") {
    div.classList.add("user");
  } else if (role === "bot") {
    div.classList.add("bot");
  } else if (role === "evaluation") {
    div.classList.add("evaluation");
  }

  div.innerText = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
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

  add("bot", "📄 Uploading & indexing document...");

  const res = await fetch("/index", {
    method: "POST",
    body: form,
  });

  const data = await res.json();

  add("bot", "✅ Document indexed successfully");
}

// ------------------------
// ASK QUESTION
// ------------------------
async function ask() {
  const q = questionInput.value.trim();
  if (!q) return;

  add("user", q);
  questionInput.value = "";

  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: q }),
  });

  const data = await res.json();

  add("bot", data.answer);

  if (data.evaluation) {
    add(
      "evaluation",
      `📊 Evaluation
Faithfulness: ${data.evaluation.faithfulness_score}
Hallucination: ${data.evaluation.hallucination}

${data.evaluation.explanation}`
    );
  }
}

// ENTER key support
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask();
});
