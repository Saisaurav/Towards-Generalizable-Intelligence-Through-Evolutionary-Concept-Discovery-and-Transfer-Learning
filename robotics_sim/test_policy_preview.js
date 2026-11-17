const fs = require('fs');
const path = require('path');
const Sim = require('../llm_game_platform/web/robotics_sim.js');
const root = __dirname;
const cases = [
  { kind: 'turtlebot', file: 'turtlebot.json', cfg: {rnoise:.05,rdelay:2,rlag:1.5}, threshold:.70 },
  { kind: 'arm2link', file: 'arm2link.json', cfg: {rnoise:.05,rdelay:2,rlag:1.5,feat:3,fk:.25}, threshold:.70 }
];
for (const test of cases) {
  const json = JSON.parse(fs.readFileSync(path.join(root,'policies',test.file),'utf8'));
  const policy = new Sim.Policy(json);
  if (policy.inputIds.length !== json.obs_names.length) throw new Error(`${test.kind}: observation metadata mismatch`);
  let wins=0, reward=0;
  for(let ep=0;ep<500;ep++) {
    const env=Sim.create(test.kind,test.cfg,ep+1234);let done=false,guard=0;
    while(!done && guard++<500) done=env.step(policy.act(env.obs()));
    const t=env.telemetry();
    if(t.status==='GOAL REACHED'||t.status==='TARGET REACHED')wins++;
    reward+=t.reward;
  }
  const success=wins/500, meanReward=reward/500;
  console.log(`${test.kind}: browser mirror success ${(100*success).toFixed(1)}%, mean return ${meanReward.toFixed(3)}`);
  if(success<test.threshold) throw new Error(`${test.kind}: preview policy below sanity threshold`);
}
