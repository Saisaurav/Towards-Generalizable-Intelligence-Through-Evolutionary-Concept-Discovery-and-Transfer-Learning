import json
from pathlib import Path
from neat_policy_runtime import NeatPolicy

ROOT = Path(__file__).resolve().parents[1]
for kind in ("turtlebot", "arm2link"):
    path = ROOT / "policies" / f"{kind}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    policy = NeatPolicy(path)
    observation = [((i % 7) - 3) / 3 for i in range(len(data["input_ids"]))]
    action = policy.act(observation)
    assert len(action) == 2, (kind, action)
    assert all(-1.0 <= x <= 1.0 for x in action), (kind, action)
    print(f"{kind}: {len(observation)} inputs -> {action}")
