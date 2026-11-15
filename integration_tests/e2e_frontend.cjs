// Browser-level end-to-end test of the merged frontend (runs the page's real JavaScript in jsdom).
//   NODE_PATH=<dir with jsdom> node e2e_frontend.cjs http://127.0.0.1:PORT
// Server must be started with e2e_stub_server.py. Every check prints PASS/FAIL; exit code 1 on any FAIL.
const { JSDOM } = require("jsdom");
const BASE = process.argv[2];
const results = [];
const check = (name, ok, detail = "") => { results.push(ok); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  -- " + detail : ""}`); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(fn, ms, what) { const end = Date.now() + ms; while (Date.now() < end) { try { const v = fn(); if (v) return v; } catch (e) {} await sleep(250); } throw new Error("timed out waiting for: " + what); }

(async () => {
  const requests = [];
  const dom = await JSDOM.fromURL(BASE + "/", {
    runScripts: "dangerously", resources: "usable", pretendToBeVisual: true,
    beforeParse(window) {
      window.fetch = (u, o) => { const url = new URL(u, BASE).href; requests.push(`${(o && o.method) || "GET"} ${new URL(url).pathname}`); return fetch(url, o); };
    },
  });
  const { window } = dom; const $ = (id) => window.document.getElementById(id);
  await until(() => window.document.readyState === "complete", 10000, "page load");
  const errors = []; window.addEventListener("error", (e) => errors.push(String(e.message)));

  // ---- Frontend: structure & navigation
  check("page loads with generated-env tab active", $("tab-env").classList.contains("active") && !$("env-mode").hidden);
  check("research link present in nav", $("tab-research") && $("tab-research").getAttribute("href") === "/research");
  $("tab-legacy").click(); await sleep(100);
  check("tab switch to legacy grid mode works", !$("legacy-mode").hidden && $("env-mode").hidden);
  $("tab-env").click(); await sleep(100);
  check("tab switch back to generated mode works", !$("env-mode").hidden && $("legacy-mode").hidden);

  // ---- Game builder: describe a game and submit
  $("env-prompt").value = "Tic-tac-toe on a 3x3 board for two players, X moves first, three in a row wins";
  $("env-train-opp").value = "random";
  $("env-form").dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
  await until(() => requests.some((r) => r === "POST /api/envs"), 10000, "POST /api/envs");
  check("generation request reached backend (POST /api/envs)", true);

  // ---- Environment generation + visualization
  await until(() => $("env-title").textContent === "Tic-Tac-Toe", 60000, "title from generated SPEC");
  check("generated SPEC title shown", $("env-title").textContent === "Tic-Tac-Toe");
  await until(() => /generated and validated/i.test($("env-gen-status").textContent), 60000, "generation status");
  check("generation status says generated and validated", true, $("env-gen-status").textContent);
  check("sandbox validation report displayed", /Validated in a sandbox on \d+ random episodes/.test($("env-validation").textContent), $("env-validation").textContent.slice(0, 90));
  await until(() => requests.some((r) => r === "POST /api/envs/" + (window.__id || "") + "/reset") || requests.some((r) => /POST \/api\/envs\/[0-9a-f]{32}\/reset/.test(r)), 30000, "auto reset (environment initialisation)");
  check("environment initialised via /reset", true);
  await until(() => $("env-scene").querySelectorAll("svg .clickable").length === 9, 30000, "9 clickable board cells");
  check("board scene rendered with 9 legal clickable cells (initial observation/legal actions)", true);

  // ---- Brain training in the generated environment (real C++ NEAT+PPO via relay)
  await until(() => /Training complete/.test($("env-training").textContent) || /Training failed/.test($("env-training").textContent), 240000, "training to finish");
  check("training completed (not failed)", /Training complete/.test($("env-training").textContent), $("env-training").textContent);
  check("per-generation chart drawn from training_history (polylines)", $("env-chart").querySelectorAll("polyline").length === 2);
  check("population info line populated", /gen \d+ \| \d+ env steps \| \d+ species/.test($("env-popinfo").textContent), $("env-popinfo").textContent);
  const concepts = $("env-concepts").textContent.split("\n").filter(Boolean);
  check("concept list from V8 concept bank displayed", concepts.length > 0 && /^Concepts available to the learner \(\d+\)/.test($("env-concept-summary").textContent), `${concepts.length} concepts; e.g. ${concepts[0]}`);
  const id = (await (await fetch(BASE + "/api/envs")).json()).environments[0].id;
  const status = await (await fetch(`${BASE}/api/envs/${id}/status`)).json();
  check("status API reports ready brain + held-out return", status.brain_ready === true && typeof status.training_heldout === "number", `heldout=${status.training_heldout}`);
  check("concepts reaching the brain are plain V8 bank expressions (no NaN-primitive concepts)", status.concepts.every((c) => !/board_|transition_|action_wins|action_blocks|state_boundary_margin/.test(c)));

  // ---- Runtime loop: submit a human action, then let the trained brain act
  $("env-brain").disabled === false || await until(() => !$("env-brain").disabled, 10000, "brain button enabled");
  check("'brain plays' control enabled after training", !$("env-brain").disabled);
  const cells = $("env-scene").querySelectorAll("svg .clickable"); const before = $("env-scene").querySelectorAll("svg .clickable").length;
  cells[4].dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await until(() => requests.some((r) => /POST \/api\/envs\/[0-9a-f]{32}\/action/.test(r)), 10000, "POST /action");
  await until(() => $("env-scene").querySelectorAll("svg .clickable").length < before, 15000, "legal moves shrink");
  check("human action submitted; observation/legal set updated and opponent replied", $("env-scene").querySelectorAll("svg .clickable").length <= before - 1, `${before} -> ${$("env-scene").querySelectorAll("svg .clickable").length} legal`);
  const legalBefore = $("env-scene").querySelectorAll("svg .clickable").length;
  $("env-brain").click();
  await until(() => requests.some((r) => /POST \/api\/envs\/[0-9a-f]{32}\/agent/.test(r)), 15000, "POST /agent (brain acts)");
  await until(() => $("env-scene").querySelectorAll("svg .clickable").length !== legalBefore || /won|lost|drew/.test($("env-hud").textContent), 20000, "brain move applied");
  check("trained brain acted through Python policy inference (no error)", $("env-error").textContent === "", $("env-error").textContent || "no error");
  // let the game play out to a terminal state
  await until(() => /won|lost|drew/.test($("env-hud").textContent), 60000, "episode termination");
  check("episode terminates and result displayed in HUD", true, $("env-hud").textContent);
  window.close();

  // ---- Research viewer retained from V8
  const r = await fetch(BASE + "/research"); const html = await r.text();
  check("/research page served", r.status === 200 && /research view/i.test(html));
  const c = await fetch(BASE + "/data/concepts.json"); const cj = await c.json();
  check("/data/concepts.json served and parseable", c.status === 200 && Array.isArray(cj.concepts) && cj.concepts.length > 0, `${cj.concepts.length} concepts`);
  const e = await fetch(BASE + "/data/experiments.json"); const ej = await e.json();
  check("/data/experiments.json served and parseable", e.status === 200 && Array.isArray(ej.groups) && ej.groups.length > 0, `${ej.groups.length} run groups`);
  check("no uncaught JS errors in page", errors.length === 0, errors.join("; "));
  console.log(`\n${results.filter(Boolean).length}/${results.length} checks passed`); process.exit(results.every(Boolean) ? 0 : 1);
})().catch((e) => { console.log("FAIL  harness error: " + e.message); process.exit(1); });
