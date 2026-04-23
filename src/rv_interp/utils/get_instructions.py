import json
from libero.libero import benchmark

def get_all_suite_instructions():
    # Retrieve the dictionary of all available benchmarks/suites
    benchmark_dict = benchmark.get_benchmark_dict()

    # Common suites: 'libero_spatial', 'libero_object', 'libero_goal', 
    # 'libero_90' (Training part of LIBERO-100), 'libero_10' (Test part)

    all_data = {}


    for suite_name, suite_class in benchmark_dict.items():
        if suite_name == 'libero_100':
            continue
        print(f"Processing suite: {suite_name}...")
        
        task_suite = suite_class()
        num_tasks = task_suite.get_num_tasks()
        
        for i in range(num_tasks):
            task = task_suite.get_task(i)
            all_data[task.language] = task.language
            
    return all_data

all_suites_info = get_all_suite_instructions()

with open("libero_suite_instructions.json", "w") as f:
    json.dump(all_suites_info, f, indent=4)

print("\nExtraction complete. Total suites processed:", len(all_suites_info))
