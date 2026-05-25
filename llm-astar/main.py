import openai
import os
import glob
import argparse
from dotenv import load_dotenv
import json
from datetime import datetime
from collections import defaultdict
from llmastar.pather import AStar, LLMVLMRunner

if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
api_key = os.getenv('OPENAI_API_KEY')
if not api_key or api_key == 'your_new_openai_api_key_here':
    print("Error: OPENAI_API_KEY not found or not set in .env file.")
    print(f".env file path: {env_path}")
    print("Please update OPENAI_API_KEY in your .env file with a valid API key.")
    exit(1)
openai.api_key = api_key
print(f"API key loaded: {api_key[:10]}...{api_key[-4:]}")

def run_all_dataset(llm='deepseek', vlm='llama-4', alpha_decay=0.9, dataset_name='Dataset_demo', prompt='repe'):
    """
    Execute all queries in the specified dataset using A*, LLM-A*, and VLM-A* methods.
    Returns a dictionary containing the results for each query.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(os.path.dirname(current_dir), dataset_name)
    query_files = glob.glob(f"{dataset_path}/*/query_*.json")
    query_files.sort()
    print(f"Found {len(query_files)} query files in dataset.")

    all_results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_queries": len(query_files),
        "results": {},
        "dataset_name": dataset_name
    }

    runner = LLMVLMRunner(llm=llm, vlm=vlm, prompt=prompt, alpha_decay=alpha_decay)

    for i, query_file in enumerate(query_files, 1):
        parts = query_file.replace(dataset_path, "").strip("/").split("/")
        map_name = parts[0]
        query_name = parts[1].replace(".json", "")
        test_id = f"{map_name}_{query_name}"

        print(f"\n{'='*60}")    
        print(f"Processing {i}/{len(query_files)}: {test_id}")
        print(f"{'='*60}")

        try:
            with open(query_file, 'r') as f:
                query = json.load(f)
        except Exception as e:
            print(f"Error reading {query_file}: {e}")
            continue

        test_results = {
            "map_name": map_name,
            "query_name": query_name,
            "query_file": query_file,
            "query": query,
            "methods": {}
        }

        print("\nRunning A*...")
        try:
            astar_result = AStar().searching(query=query, filepath='temp_astar.png', no_plot=True)
            test_results["methods"]["A*"] = extract_result_fields(astar_result)
            print(f"   A* completed: ops={test_results['methods']['A*']['operation']}, storage={test_results['methods']['A*']['storage']}, length={test_results['methods']['A*']['length']}")
        except Exception as e:
            print(f"   A* failed: {e}")
            test_results["methods"]["A*"] = {"error": str(e)}

        print("\nRunning LLM/VLM methods...")
        try:
            llm_vlm_results = runner.run_all_four(query, 'temp_method', no_plot=True)
            for method_name, result in llm_vlm_results.items():
                test_results["methods"][method_name] = extract_result_fields(result)
                if result:
                    print(f"   {method_name}: ops={test_results['methods'][method_name]['operation']}, storage={test_results['methods'][method_name]['storage']}, length={test_results['methods'][method_name]['length']}")
                else:
                    print(f"   {method_name}: failed")
        except Exception as e:
            print(f"   LLM/VLM methods failed: {e}")
            test_results["methods"]["LLM/VLM_ERROR"] = {"error": str(e)}

        all_results["results"][test_id] = test_results
        print(f"\nCompleted {test_id}")

    print(f"\n{'='*60}")
    print("DATASET EVALUATION COMPLETED")
    print(f"Total queries processed: {len(query_files)}")
    print(f"{'='*60}")

    return all_results

def extract_result_fields(result):
    """
    Extracts relevant fields from the result dictionary for evaluation.
    Returns a dictionary with operation, storage, length, path_found, and checkpoints.
    """
    if result is None:
        return {"operation": None, "storage": None, "length": None, "path_found": None, "checkpoints": None, "error": "Result is None"}
    checkpoints = None
    if 'llm_output' in result:
        checkpoints = result.get('llm_output', None)
    elif 'checkpoints' in result:
        checkpoints = result.get('checkpoints', None)
    return {
        "operation": result.get('operation', None),
        "storage": result.get('storage', None),
        "length": result.get('length', None),
        "path_found": result.get('path_found', None),
        "checkpoints": checkpoints
    }

def analyze_dataset_results(data):
    """
    Analyzes the results of all queries in the dataset.
    Computes the relative performance of each method compared to A* (baseline = 100%).
    Returns a summary table with average percentages for each method and metric.
    """
    metrics = ['operation', 'storage', 'length']
    method_names = ['LLM-A*', 'LLM-A* (Adaptive)', 'VLM-A*', 'VLM-A* (Adaptive)']
    dataset_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Determine if this is Complex_Dataset or Resolution_Dataset for per-level summary
    is_complex_dataset = False
    if "dataset_name" in data and (data["dataset_name"] == "Complex_Dataset" or data["dataset_name"] == "Resolution_Dataset"):
        is_complex_dataset = True

    # For Complex_Dataset, extract level from map_name (should be level_1, level_2, ...)
    for query_id, query_data in data['results'].items():
        methods = query_data.get('methods', {})
        astar_result = methods.get('A*', {})
        if not astar_result or 'error' in astar_result:
            continue
        map_name = query_data['map_name']
        if is_complex_dataset:
            # map_name is expected to be like "level_1", "level_2", etc.
            level = map_name
        else:
            level = map_name
        astar_metrics = {}
        for metric in metrics:
            astar_value = astar_result.get(metric)
            if astar_value is not None and astar_value > 0:
                astar_metrics[metric] = astar_value
            else:
                astar_metrics[metric] = None
        for method_name in method_names:
            method_result = methods.get(method_name, {})
            if method_result and 'error' not in method_result:
                for metric in metrics:
                    astar_val = astar_metrics[metric]
                    method_val = method_result.get(metric)
                    if astar_val is not None and method_val is not None and astar_val > 0:
                        percentage = (method_val / astar_val) * 100
                        dataset_results[level][method_name][metric].append(percentage)

    dataset_averages = defaultdict(lambda: defaultdict(dict))
    for level in dataset_results:
        for method_name in method_names:
            for metric in metrics:
                values = dataset_results[level][method_name][metric]
                if values:
                    avg_percentage = sum(values) / len(values)
                    dataset_averages[level][method_name][metric] = avg_percentage
                else:
                    dataset_averages[level][method_name][metric] = None

    overall_averages = defaultdict(dict)
    for method_name in method_names:
        for metric in metrics:
            dataset_avgs = []
            for level in dataset_averages:
                if dataset_averages[level][method_name][metric] is not None:
                    dataset_avgs.append(dataset_averages[level][method_name][metric])
            if dataset_avgs:
                overall_avg = sum(dataset_avgs) / len(dataset_avgs)
                overall_averages[method_name][metric] = overall_avg
            else:
                overall_averages[method_name][metric] = None

    if is_complex_dataset:
        print(f"\n{'='*80}")
        print("SUMMARY TABLE BY LEVEL (Complex_Dataset)")
        print("="*80)
        print(f"{'Level':<10} {'Method':<20} {'Operations':<12} {'Storage':<12} {'Length':<12}")
        print("-"*70)
        for level in sorted(dataset_averages.keys(), key=lambda x: int(x.split('_')[-1]) if x.startswith('level_') else x):
            print(f"{level:<10} {'A*':<20} {'100.0%':<12} {'100.0%':<12} {'100.0%':<12}")
            for method_name in method_names:
                row = f"{level:<10} {method_name:<20}"
                for metric in metrics:
                    avg = dataset_averages[level][method_name][metric]
                    if avg is not None:
                        row += f"{avg:6.1f}%{'':<6}"
                    else:
                        row += f"{'N/A':<12}"
                print(row)
        print(f"\n{'='*80}")
        print("OVERALL AVERAGE (All Levels)")
        print("="*80)
        print(f"{'Method':<20} {'Operations':<12} {'Storage':<12} {'Length':<12}")
        print("-"*60)
        print(f"{'A*':<20} {'100.0%':<12} {'100.0%':<12} {'100.0%':<12}")
        for method_name in method_names:
            row = f"{method_name:<20}"
            for metric in metrics:
                avg = overall_averages[method_name][metric]
                if avg is not None:
                    row += f"{avg:6.1f}%{'':<6}"
                else:
                    row += f"{'N/A':<12}"
            print(row)
    else:
        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print("="*80)
        print(f"{'Method':<20} {'Operations':<12} {'Storage':<12} {'Length':<12}")
        print("-"*60)
        print(f"{'A*':<20} {'100.0%':<12} {'100.0%':<12} {'100.0%':<12}")
        for method_name in method_names:
            row = f"{method_name:<20}"
            for metric in metrics:
                avg = overall_averages[method_name][metric]
                if avg is not None:
                    row += f"{avg:6.1f}%{'':<6}"
                else:
                    row += f"{'N/A':<12}"
            print(row)

    # Prepare summary_table for return
    summary_table = {}
    if is_complex_dataset:
        summary_table['A*'] = {}
        for level in sorted(dataset_averages.keys(), key=lambda x: int(x.split('_')[-1]) if x.startswith('level_') else x):
            summary_table[level] = {}
            summary_table[level]['A*'] = {'operations': 100.0, 'storage': 100.0, 'length': 100.0}
            for method_name in method_names:
                summary_table[level][method_name] = {}
                for metric in metrics:
                    avg = dataset_averages[level][method_name][metric]
                    if avg is not None:
                        if metric == 'operation':
                            summary_table[level][method_name]['operations'] = round(avg, 1)
                        elif metric == 'storage':
                            summary_table[level][method_name]['storage'] = round(avg, 1)
                        elif metric == 'length':
                            summary_table[level][method_name]['length'] = round(avg, 1)
                    else:
                        if metric == 'operation':
                            summary_table[level][method_name]['operations'] = None
                        elif metric == 'storage':
                            summary_table[level][method_name]['storage'] = None
                        elif metric == 'length':
                            summary_table[level][method_name]['length'] = None
        # Also add overall
        summary_table['overall'] = {'A*': {'operations': 100.0, 'storage': 100.0, 'length': 100.0}}
        for method_name in method_names:
            summary_table['overall'][method_name] = {}
            for metric in metrics:
                avg = overall_averages[method_name][metric]
                if avg is not None:
                    if metric == 'operation':
                        summary_table['overall'][method_name]['operations'] = round(avg, 1)
                    elif metric == 'storage':
                        summary_table['overall'][method_name]['storage'] = round(avg, 1)
                    elif metric == 'length':
                        summary_table['overall'][method_name]['length'] = round(avg, 1)
                else:
                    if metric == 'operation':
                        summary_table['overall'][method_name]['operations'] = None
                    elif metric == 'storage':
                        summary_table['overall'][method_name]['storage'] = None
                    elif metric == 'length':
                        summary_table['overall'][method_name]['length'] = None
    else:
        summary_table = {
            'A*': {
                'operations': 100.0,
                'storage': 100.0,
                'length': 100.0
            }
        }
        for method_name in method_names:
            summary_table[method_name] = {}
            for metric in metrics:
                avg = overall_averages[method_name][metric]
                if avg is not None:
                    if metric == 'operation':
                        summary_table[method_name]['operations'] = round(avg, 1)
                    elif metric == 'storage':
                        summary_table[method_name]['storage'] = round(avg, 1)
                    elif metric == 'length':
                        summary_table[method_name]['length'] = round(avg, 1)
                else:
                    if metric == 'operation':
                        summary_table[method_name]['operations'] = None
                    elif metric == 'storage':
                        summary_table[method_name]['storage'] = None
                    elif metric == 'length':
                        summary_table[method_name]['length'] = None

    analysis_results = {
        'summary_table': summary_table,
        'baseline': 'A* = 100%',
        'timestamp': data['timestamp']
    }
    return analysis_results

def parse_arguments():
    """
    Parses command line arguments for the evaluation script.
    """
    parser = argparse.ArgumentParser(description='Run LLM-A* and VLM-A* pathfinding evaluation')
    parser.add_argument('--llm', type=str, default='gpt',
                       choices=['gpt', 'llama3_fpt', 'qwen', 'deepseek'],
                       help='LLM model to use (default: gpt)')
    parser.add_argument('--vlm', type=str, default='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
                       choices=['meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
                               'google/gemma-3n-E4B-it',
                               'Qwen/Qwen2.5-VL-72B-Instruct'],
                       help='VLM model to use (default: meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)')
    parser.add_argument('--alpha_decay', type=float, default=0.9,
                       help='Alpha decay factor for adaptive heuristic (default: 0.9)')
    parser.add_argument('--dataset_name', type=str, default='Dataset_demo',
                       choices=['Dataset', 'Dataset_demo', 'Complex_Dataset', 'Resolution_Dataset'],  
                       help='Dataset directory to use (default: Dataset_demo)')
    parser.add_argument('--prompt', type=str, default='repe',
                       choices=['standard', 'cot', 'repe'],
                       help='Prompt type to use (default: repe)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Starting evaluation with configuration:")
    print(f"   LLM: {args.llm}")
    print(f"   VLM: {args.vlm}")
    print(f"   Alpha Decay: {args.alpha_decay}")
    print(f"   Dataset: {args.dataset_name}")
    print(f"   Prompt: {args.prompt}")
    print("="*60)

    results = run_all_dataset(
        llm=args.llm,
        vlm=args.vlm,
        alpha_decay=args.alpha_decay,
        dataset_name=args.dataset_name,
        prompt=args.prompt
    )

    analysis = analyze_dataset_results(results)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")

    temp_files = glob.glob("temp_*.png")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except:
            pass