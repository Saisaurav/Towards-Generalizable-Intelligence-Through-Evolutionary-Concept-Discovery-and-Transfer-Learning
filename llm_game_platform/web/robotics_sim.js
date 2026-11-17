/*
 * Browser-side robotics surrogate and NEAT policy evaluator.
 * Dynamics/observation conventions mirror src/robot_envs.h. This is a browser
 * visualization of the trained surrogate, not a live Gazebo renderer.
 */
(function (root) {
  'use strict';
  const OUT_BASE = (1 << 24) + 1, HID_BASE = 1 << 25;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const wrap = (a) => { while (a > Math.PI) a -= Math.PI * 2; while (a < -Math.PI) a += Math.PI * 2; return a; };
  class Rng {
    constructor(seed) { let x = BigInt(seed == null ? 1 : seed) & ((1n << 64n) - 1n); this.s=[]; for(let i=0;i<4;i++){x=(x+0x9E3779B97F4A7C15n)&((1n<<64n)-1n);let z=x;z=((z^(z>>30n))*0xBF58476D1CE4E5B9n)&((1n<<64n)-1n);z=((z^(z>>27n))*0x94D049BB133111EBn)&((1n<<64n)-1n);this.s.push((z^(z>>31n))&((1n<<64n)-1n));} }
    rotl(x,k){const mask=(1n<<64n)-1n;return ((x<<BigInt(k))|(x>>BigInt(64-k)))&mask;}
    next(){const mask=(1n<<64n)-1n,s=this.s;const result=(this.rotl((s[0]+s[3])&mask,23)+s[0])&mask,t=(s[1]<<17n)&mask;s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=this.rotl(s[3],45);return result;}
    u(){return Number(this.next()>>40n)/16777216;}
    range(a,b){return a+(b-a)*this.u();}
    below(n){return Number(this.next()%BigInt(n));}
    normal(){let u1=this.u();if(u1<1e-7)u1=1e-7;return Math.sqrt(-2*Math.log(u1))*Math.cos(6.2831853*this.u());}
  }
  class Policy {
    constructor(json) {
      this.json = json;
      if (!json || !Array.isArray(json.nodes) || !Array.isArray(json.connections) || !Array.isArray(json.input_ids)) throw new Error('Invalid policy JSON');
      this.inputIds = json.input_ids.slice(); this.actDim = Number(json.act_dim || json.action_dim || 0);
      if (!this.actDim) throw new Error('Policy is not a continuous-action robot policy');
      const compute = json.nodes.filter(n => n.type === 2 || n.type === 3).map(n => n.id);
      const computeSet = new Set(compute), incoming = new Map(compute.map(id => [id, []]));
      for (const edge of json.connections) if (edge.enabled && computeSet.has(edge.to)) incoming.get(edge.to).push(edge);
      const live = new Set(), stack = [];
      for (let i = 0; i < this.actDim; i++) { const id = OUT_BASE + i; if (!computeSet.has(id)) throw new Error('Policy output node missing'); live.add(id); stack.push(id); }
      while (stack.length) {
        const id = stack.pop();
        for (const edge of incoming.get(id) || []) if (computeSet.has(edge.from) && !live.has(edge.from)) { live.add(edge.from); stack.push(edge.from); }
      }
      const indegree = new Map(), adj = new Map();
      for (const id of live) { indegree.set(id, 0); adj.set(id, []); }
      for (const id of live) for (const edge of incoming.get(id) || []) if (live.has(edge.from)) { indegree.set(id, indegree.get(id) + 1); adj.get(edge.from).push(id); }
      const queue = compute.filter(id => live.has(id) && indegree.get(id) === 0), order = [];
      while (queue.length) { const id = queue.shift(); order.push(id); for (const to of adj.get(id) || []) { indegree.set(to, indegree.get(to) - 1); if (indegree.get(to) === 0) queue.push(to); } }
      if (order.length !== live.size) throw new Error('Policy contains a cyclic active topology');
      this.order = order; this.incoming = incoming; this.values = new Map();
      this.nodes = new Map(json.nodes.map(n => [n.id, n]));
    }
    act(obs) {
      if (obs.length !== this.inputIds.length) throw new Error(`Observation mismatch: policy expects ${this.inputIds.length}, simulator provides ${obs.length}`);
      this.values.clear(); this.inputIds.forEach((id, i) => this.values.set(id, Number(obs[i]) || 0)); this.values.set(1 << 24, 1);
      for (const id of this.order) {
        let sum = 0;
        for (const edge of this.incoming.get(id) || []) sum += (this.values.get(edge.from) || 0) * edge.weight;
        this.values.set(id, this.nodes.get(id).type === 3 ? Math.tanh(sum) : sum);
      }
      return Array.from({ length: this.actDim }, (_, i) => clamp(this.values.get(OUT_BASE + i) || 0, -1, 1));
    }
  }
  class Channel {
    constructor(cfg, rng) { this.cfg = cfg || {}; this.rng = rng; this.begin(); }
    begin() {
      this.noise = Number(this.cfg.noise || 0); this.lag = Number(this.cfg.lag || 1); this.delay = Number(this.cfg.delay || 0);
      if (this.cfg.rnoise) this.noise = this.rng.range(0, Number(this.cfg.rnoise));
      if (Number(this.cfg.rdelay) > 0) this.delay = this.rng.below(Number(this.cfg.rdelay) + 1);
      if (Number(this.cfg.rlag) > 1) this.lag *= Math.exp(this.rng.range(-Math.log(Number(this.cfg.rlag)), Math.log(Number(this.cfg.rlag))));
      this.queue = Array.from({ length: this.delay }, () => [0, 0]);
    }
    apply(a, b) { if (this.delay <= 0) return [a, b]; this.queue.push([a, b]); return this.queue.shift(); }
  }
  class TurtleBot {
    constructor(cfg, seed) { this.cfg = cfg || {}; this.rng = new Rng(seed); this.horizon = Number(this.cfg.horizon || 300); this.reset(); }
    reset() {
      const r = this.rng; this.channel = new Channel(this.cfg, r); this.orng = new Rng(r.next()); this.t = 0; this.reached = false; this.crashed = false;
      this.x = r.range(-1.7, 1.7); this.y = r.range(-1.7, 1.7); this.th = r.range(-Math.PI, Math.PI);
      do { this.gx = r.range(-1.7, 1.7); this.gy = r.range(-1.7, 1.7); } while (Math.hypot(this.gx - this.x, this.gy - this.y) < 1);
      this.obstacles = []; const n = 3 + r.below(4);
      for (let i = 0; i < n; i++) { let placed = false; for (let tries = 0; tries < 50; tries++) { const radius = r.range(.1, .25), x = r.range(-1.7, 1.7), y = r.range(-1.7, 1.7); if (Math.hypot(x-this.x,y-this.y) > radius+.105+.25 && Math.hypot(x-this.gx,y-this.gy) > radius+.15+.2) { this.obstacles.push({x,y,r:radius}); placed = true; break; } } if (!placed) this.obstacles.push({x:9,y:9,r:.1}); }
      this.v = this.w = this.pa0 = this.pa1 = 0; this.dprev = Math.hypot(this.gx-this.x,this.gy-this.y); this.reward = 0; this.done = false; this.lastAction = [0,0];
    }
    cast(angle) {
      const dx = Math.cos(angle), dy = Math.sin(angle); let best = 3.5;
      for (const o of this.obstacles) { const fx=this.x-o.x, fy=this.y-o.y, b=fx*dx+fy*dy, c=fx*fx+fy*fy-o.r*o.r, disc=b*b-c; if (disc>=0) { const tt=-b-Math.sqrt(disc); if (tt>=0 && tt<best) best=tt; } }
      if (dx>1e-6) best=Math.min(best,(2-this.x)/dx); else if(dx< -1e-6) best=Math.min(best,(-2-this.x)/dx);
      if (dy>1e-6) best=Math.min(best,(2-this.y)/dy); else if(dy< -1e-6) best=Math.min(best,(-2-this.y)/dy);
      return Math.max(0,best);
    }
    obs() {
      const n=this.channel.noise, r=this.orng, out=[];
      for(let i=0;i<24;i++) out.push(clamp(this.cast(this.th+Math.PI*2*i/24)+n*r.normal(),0,3.5)/3.5);
      const dx=this.gx-this.x+n*r.normal(), dy=this.gy-this.y+n*r.normal(), d=Math.hypot(dx,dy), b=wrap(Math.atan2(dy,dx)-this.th);
      out.push(Math.min(d,4)/4,Math.sin(b),Math.cos(b),this.v/.22,this.w/2,this.pa0,this.pa1,this.t/this.horizon);
      return out;
    }
    step(action) {
      if(this.done) return true;
      const a0=clamp(action[0]||0,-1,1),a1=clamp(action[1]||0,-1,1); this.pa0=a0;this.pa1=a1;this.lastAction=[a0,a1];
      const c=this.channel.apply(a0,a1), al=1-Math.exp(-.1/(.2*this.channel.lag));
      this.v+=al*(c[0]*.22-this.v);this.w+=al*(c[1]*2-this.w);this.x+=this.v*Math.cos(this.th)*.1;this.y+=this.v*Math.sin(this.th)*.1;this.th=wrap(this.th+this.w*.1);this.t++;
      const d=Math.hypot(this.gx-this.x,this.gy-this.y), r=(this.dprev-d)-.002;this.dprev=d;this.reward+=r;
      if(d<.15){this.reached=true;this.done=true;this.reward+=2;}
      let hit=Math.abs(this.x)>2-.105||Math.abs(this.y)>2-.105;
      for(const o of this.obstacles) if(Math.hypot(this.x-o.x,this.y-o.y)<o.r+.105) hit=true;
      if(hit){this.crashed=true;this.done=true;this.reward-=1;}
      if(this.t>=this.horizon)this.done=true;
      return this.done;
    }
    telemetry(){return {x:this.x,y:this.y,heading:this.th,linear:this.v,angular:this.w,goal:Math.hypot(this.gx-this.x,this.gy-this.y),lidar:Math.min(...Array.from({length:24},(_,i)=>this.cast(this.th+Math.PI*2*i/24))),reward:this.reward,step:this.t,horizon:this.horizon,status:this.reached?'GOAL REACHED':this.crashed?'COLLISION':this.done?'TIME LIMIT':'RUNNING'};}
  }
  class Arm2Link {
    constructor(cfg, seed) { this.cfg=cfg||{};this.rng=new Rng(seed);this.horizon=Number(this.cfg.horizon||100);this.feat=Number(this.cfg.feat||0);this.fk=Number(this.cfg.fk||.25);this.reset(); }
    fkPos(a,b){return {x:.30*Math.cos(a)+.25*Math.cos(a+b),y:.30*Math.sin(a)+.25*Math.sin(a+b)};}
    reset(){const r=this.rng;this.channel=new Channel(this.cfg,r);this.orng=new Rng(r.next());this.q1=r.range(-1.5,1.5);this.q2=r.range(-1.5,1.5);const target=this.fkPos(r.range(-2.5,2.5),r.range(-2.4,2.4));this.tx=target.x;this.ty=target.y;this.d1=this.d2=this.pa0=this.pa1=this.lastSpeed=0;this.t=0;this.ok=false;this.done=false;this.reward=0;this.qh1=this.qh2=this.qf1=this.qf2=0;this.qinit=false;this.lastAction=[0,0]; }
    obs(){const n=this.channel.noise,r=this.orng;let a=this.q1+n*r.normal(),b=this.q2+n*r.normal();if(this.feat>=3){const p1=this.qinit?this.qh1+this.pa0*2*.05:a,p2=this.qinit?this.qh2+this.pa1*2*.05:b;a=this.qf1=p1+this.fk*(a-p1);b=this.qf2=p2+this.fk*(b-p2);}const ee=this.fkPos(a,b),ex=this.tx-ee.x,ey=this.ty-ee.y,out=[Math.sin(a),Math.cos(a),Math.sin(b),Math.cos(b),this.d1/2+5*n*r.normal(),this.d2/2+5*n*r.normal(),ex/.55,ey/.55,this.pa0,this.pa1];if(this.feat>=1)out.push(clamp(ex/.08,-1,1),clamp(ey/.08,-1,1));if(this.feat>=2){const s1=Math.sin(a),c1=Math.cos(a),s12=Math.sin(a+b),c12=Math.cos(a+b),j11=-.30*s1-.25*s12,j12=-.25*s12,j21=.30*c1+.25*c12,j22=.25*c12;out.push(clamp((j11*ex+j21*ey)/.05,-1,1),clamp((j12*ex+j22*ey)/.05,-1,1));}if(this.cfg.tfeat!==0)out.push(this.t/this.horizon);return out;}
    step(action){if(this.done)return true;if(this.feat>=3){this.qh1=this.qf1;this.qh2=this.qf2;this.qinit=true;}const a0=clamp(action[0]||0,-1,1),a1=clamp(action[1]||0,-1,1);this.pa0=a0;this.pa1=a1;this.lastAction=[a0,a1];const c=this.channel.apply(a0,a1),al=1-Math.exp(-.05/(.1*this.channel.lag));this.d1+=al*(c[0]*2-this.d1);this.d2+=al*(c[1]*2-this.d2);this.q1+=this.d1*.05;this.q2+=this.d2*.05;if(this.q1>3||this.q1< -3){this.q1=clamp(this.q1,-3,3);this.d1=0;}if(this.q2>2.6||this.q2< -2.6){this.q2=clamp(this.q2,-2.6,2.6);this.d2=0;}const ee=this.fkPos(this.q1,this.q2),dist=Math.hypot(this.tx-ee.x,this.ty-ee.y);this.lastSpeed=Math.hypot(this.d1,this.d2)/2;this.t++;this.ok=dist<.03&&this.lastSpeed<.1;this.reward+=-.1*dist/.3-.01*(a0*a0+a1*a1);if(this.t>=this.horizon)this.done=true;return this.done;}
    telemetry(){const ee=this.fkPos(this.q1,this.q2);return {x:ee.x,y:ee.y,q1:this.q1,q2:this.q2,goal:Math.hypot(this.tx-ee.x,this.ty-ee.y),linear:this.lastSpeed,angular:0,reward:this.reward,step:this.t,horizon:this.horizon,status:this.done&&this.ok?'TARGET REACHED':this.done?'TIME LIMIT':this.ok?'TARGET IN TOLERANCE':'RUNNING'};}
  }
  function create(kind,cfg,seed){return kind==='turtlebot'?new TurtleBot(cfg,seed):new Arm2Link(cfg,seed);}
  root.RobotSim={Policy,Rng,TurtleBot,Arm2Link,create};
  if(typeof module!=='undefined'&&module.exports)module.exports=root.RobotSim;
})(typeof window!=='undefined'?window:globalThis);
