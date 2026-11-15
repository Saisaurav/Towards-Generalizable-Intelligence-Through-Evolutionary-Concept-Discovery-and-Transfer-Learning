// Browser-level smoke test for manual frame playback in real-time games.
//   NODE_PATH=<dir with jsdom> node e2e_realtime.cjs http://127.0.0.1:PORT
// Server must be started with e2e_stub_server.py using the snake fixture.
const { JSDOM } = require("jsdom");
const BASE = process.argv[2];
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function until(fn, ms, what) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    try { const value = fn(); if (value) return value; } catch (_) {}
    await sleep(100);
  }
  throw new Error("timed out waiting for: " + what);
}

(async () => {
  const requests = [];
  const dom = await JSDOM.fromURL(BASE + "/", {
    runScripts: "dangerously",
    resources: "usable",
    pretendToBeVisual: true,
    beforeParse(window) {
      window.fetch = (input, options) => {
        const url = new URL(input, BASE);
        const method = options && options.method || "GET";
        requests.push({ method, path: url.pathname, body: options && options.body });
        return fetch(url, options);
      };
    },
  });
  const { window } = dom;
  const $ = (id) => window.document.getElementById(id);
  await until(() => window.document.readyState === "complete", 10000, "page load");
  const errors = [];
  window.addEventListener("error", (event) => errors.push(String(event.message)));

  $("env-prompt").value = "Snake";
  $("env-form").dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
  await until(() => $("env-title").textContent === "Snake", 60000, "generated Snake environment");
  await until(() => !$("run-training").disabled, 30000, "editable environment ready");
  if ($("env-play").hidden) throw new Error("real-time Play frames button is hidden");

  $("env-restart").click();
  await until(() => $("env-play").disabled === false, 10000, "runtime reset");
  $("env-play").click();
  await until(() => requests.filter((request) =>
    request.method === "POST" && /\/action$/.test(request.path)).length >= 2, 10000, "automatic frames");

  const beforeDirection = requests.filter((request) =>
    request.method === "POST" && /\/action$/.test(request.path)).length;
  window.document.dispatchEvent(new window.KeyboardEvent("keydown", {
    key: "ArrowUp", bubbles: true, cancelable: true,
  }));
  await until(() => requests.some((request) =>
    request.method === "POST" && /\/action$/.test(request.path) &&
    JSON.parse(request.body || "{}").action === 0), 10000, "steering input on a frame");

  $("env-play").click();
  const stoppedAt = requests.filter((request) =>
    request.method === "POST" && /\/action$/.test(request.path)).length;
  await sleep(500);
  const afterPause = requests.filter((request) =>
    request.method === "POST" && /\/action$/.test(request.path)).length;
  if (afterPause !== stoppedAt) throw new Error("frames continued after Pause frames was pressed");
  if (beforeDirection < 2) throw new Error("the Play button did not advance multiple frames");
  if (errors.length) throw new Error("uncaught frontend errors: " + errors.join("; "));

  console.log("PASS  real-time Play frames advances multiple frames");
  console.log("PASS  keyboard controls steer the running game");
  console.log("PASS  Pause frames stops automatic stepping");
  window.close();
})().catch((error) => {
  console.error("FAIL  " + error.message);
  process.exit(1);
});
