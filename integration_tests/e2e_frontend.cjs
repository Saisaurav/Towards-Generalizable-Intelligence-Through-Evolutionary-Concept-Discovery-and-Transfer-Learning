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
  check("page loads with live overview active", $("tab-overview").classList.contains("active"));
  check("research link present in header", window.document.querySelector('a[href="/research"]') !== null);
  window.document.querySelector('[data-tab-link="training"]').click(); await sleep(100);
  check("training visualizer navigation works", $("tab-training").classList.contains("active"));
  window.document.querySelector('[data-tab-link="builder"]').click(); await sleep(100);
  check("game builder navigation works", $("tab-builder").classList.contains("active"));

  // ---- Game builder: generate an editable environment without starting training
  $("env-prompt").value = "Tic-tac-toe on a 3x3 board for two players, X moves first, three in a row wins";
  $("env-train-opp").value = "random";
  $("env-form").dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
  await until(() => requests.some((r) => r === "POST /api/envs"), 10000, "POST /api/envs");
  check("generation request reached backend (POST /api/envs)", true);

  // ---- Environment generation + visualization
  await until(() => $("env-title").textContent === "Tic-Tac-Toe", 60000, "title from generated SPEC");
  check("generated SPEC title shown", $("env-title").textContent === "Tic-Tac-Toe");
  check("real-time frame control stays hidden for turn-based games", $("env-play").hidden);
  await until(() => /generated and validated/i.test($("env-gen-status").textContent), 60000, "generation status");
  check("generation status says generated and validated", true, $("env-gen-status").textContent);
  check("sandbox validation report displayed", /Sandbox validation: \d+ random episodes/.test($("env-validation").textContent), $("env-validation").textContent.slice(0, 90));
  await until(() => $("env-source-input").value.includes("Tic-Tac-Toe") && !$("run-training").disabled, 30000, "editable generated source");
  check("generated Python appears in an enabled editable textbox", $("env-source-input").value.includes("class Environment") && !$("env-source-input").disabled);
  check("generation alone does not start training", $("env-training").textContent === "Training has not started." && !requests.some((r) => /POST \/api\/envs\/[0-9a-f]{32}\/train/.test(r)));
  $("env-source-input").value = $("env-source-input").value.replace('"title": "Tic-Tac-Toe"', '"title": "Edited Tic-Tac-Toe"');
  $("run-training").click();
  await until(() => requests.some((r) => /POST \/api\/envs\/[0-9a-f]{32}\/train/.test(r)), 15000, "POST /train");
  check("Run Training submits the edited environment", true);
  await until(() => $("env-title").textContent === "Edited Tic-Tac-Toe", 30000, "runtime uses edited environment source");
  check("runtime starts from the edited source", true, $("env-title").textContent);
  await until(() => $("env-scene").querySelectorAll("svg .clickable").length === 9, 30000, "9 clickable board cells");
  check("edited environment runtime renders its legal actions", true);

  // ---- Brain training in the edited generated environment (real C++ NEAT+PPO via relay)
  await until(() => /Training complete/.test($("env-training").textContent) || /Training failed/.test($("env-training").textContent), 240000, "training to finish");
  check("training completed (not failed)", /Training complete/.test($("env-training").textContent), $("env-training").textContent);
  const id = (await (await fetch(BASE + "/api/envs")).json()).environments[0].id;
  const status = await (await fetch(`${BASE}/api/envs/${id}/status`)).json();
  const historyCount = status.training_history.length;
  const chartLines = $("training-chart").querySelectorAll("polyline").length;
  const chartPoints = $("training-chart").querySelectorAll("circle").length;
  check("fitness chart plots measured history points and lines", historyCount > 0 && chartPoints === historyCount * 2 && chartLines === (historyCount > 1 ? 2 : 0), `${historyCount} records, ${chartPoints} points, ${chartLines} lines`);
  check("population info line populated from latest generation", /generation \d+, [\d,]+ environment steps, \d+ species/.test($("env-popinfo").textContent), $("env-popinfo").textContent);
  check("active concepts shown in concept bank", Number($("concept-active-count").textContent) > 0, $("concept-active-count").textContent);
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
  // Let the active policy play to a terminal state.
  $("env-auto").click();
  await until(() => /won|lost|drew/.test($("env-hud").textContent), 60000, "episode termination");
  check("episode terminates and result displayed in HUD", true, $("env-hud").textContent);
  window.close();

  // ---- Research viewer retained from V8
  const r = await fetch(BASE + "/research"); const html = await r.text();
  check("/research page served", r.status === 200 && /research view/i.test(html));
  const c = await fetch(BASE + "/data/concepts.json"); const cj = await c.json();
  check("/data/concepts.json served and parseable", c.status === 200 && Array.isArray(cj.concepts) && cj.concepts.length > 0, `${cj.concepts.length} concepts`);
  const liveBank = await fetch(BASE + "/api/concepts"); const liveBankJson = await liveBank.json();
  check("/api/concepts serves the configured real bank with compatibility", liveBank.status === 200 && liveBankJson.concepts.length > 0 && liveBankJson.concepts.every((x) => typeof x.compatible === "boolean"), `${liveBankJson.concepts.length} bank entries`);
  const e = await fetch(BASE + "/data/experiments.json"); const ej = await e.json();
  check("/data/experiments.json served and parseable", e.status === 200 && Array.isArray(ej.groups) && ej.groups.length > 0, `${ej.groups.length} run groups`);
  check("no uncaught JS errors in page", errors.length === 0, errors.join("; "));
  console.log(`\n${results.filter(Boolean).length}/${results.length} checks passed`); process.exit(results.every(Boolean) ? 0 : 1);
})().catch((e) => { console.log("FAIL  harness error: " + e.message); process.exit(1); });
