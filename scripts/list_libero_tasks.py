from libero.libero import benchmark
import json

benchmark_dict = benchmark.get_benchmark_dict()
all_tasks = {}
for suite_name, suite_cls in benchmark_dict.items():
    if suite_name == "libero_100": continue
    ts = suite_cls()
    all_tasks[suite_name] = [{"name": t.name, "language": t.language} for t in ts.tasks]

print(json.dumps(all_tasks, indent=2))
