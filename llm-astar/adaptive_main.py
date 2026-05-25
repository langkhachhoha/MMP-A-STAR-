import os
import glob
import json
import sys
import argparse
import numpy as np
from datetime import datetime
from contextlib import redirect_stdout
from io import StringIO
from llmastar.pather import AStar
from llmastar.pather.llm_a_star import LLMAStar

def load_checkpoint_data(json_file):
    """Load checkpoint data from existing results file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded checkpoint data from: {json_file}")
        print(f"📊 Total queries in checkpoint file: {data['total_queries']}")
        return data
    except Exception as e:
        print(f"❌ Error loading checkpoint data: {e}")
        return None

def convert_checkpoints_to_tuples(checkpoints):
    """Convert checkpoints from list of lists to list of tuples"""
    if not checkpoints:
        return None
    
    try:
        # Convert [[x1, y1], [x2, y2], ...] to [(x1, y1), (x2, y2), ...]
        if isinstance(checkpoints, list) and len(checkpoints) > 0:
            if isinstance(checkpoints[0], list) and len(checkpoints[0]) >= 2:
                tuple_checkpoints = [tuple(point[:2]) for point in checkpoints]
                return tuple_checkpoints
            elif isinstance(checkpoints[0], tuple):
                return checkpoints
        return checkpoints
    except Exception as e:
        print(f"      ❌ Error converting checkpoints: {e}")
        return checkpoints

def run_adaptive_analysis(checkpoint_file, alpha_values):
    """Run adaptive analysis with different alpha_decay values using pre-computed checkpoints"""
    
    # Load checkpoint data
    checkpoint_data = load_checkpoint_data(checkpoint_file)
    if not checkpoint_data:
        return None
    
    # Find all query files in dataset
    dataset_path = os.path.join(os.getcwd(), 'Dataset')
    query_files = glob.glob(f"{dataset_path}/*/query_*.json")
    query_files.sort()  # Sort by order
    
    print(f"🔍 Found {len(query_files)} query files in dataset")
    
    # Dictionary to save results with format similar to original dataset
    all_results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_queries": len(query_files),
        "checkpoint_source": checkpoint_file,
        "alpha_values_tested": alpha_values,
        "note": "Adaptive analysis using pre-computed checkpoints, format similar to original dataset",
        "results": {}
    }
    
    for i, query_file in enumerate(query_files, 1):
        # Get map and query name from path
        parts = query_file.replace(dataset_path, "").strip("/").split("/")
        map_name = parts[0]  # map_1, map_2, etc.
        query_name = parts[1].replace(".json", "")  # query_1, query_2, etc.
        test_id = f"{map_name}_{query_name}"
        
        print(f"\n{'='*60}")
        print(f"📊 Processing {i}/{len(query_files)}: {test_id}")
        print(f"{'='*60}")
        
        # Read query from file
        try:
            with open(query_file, 'r') as f:
                query = json.load(f)
        except Exception as e:
            print(f"❌ Error reading {query_file}: {e}")
            continue
        
        # Get checkpoint data for this specific query
        query_results = checkpoint_data['results'].get(test_id)
        if not query_results:
            print(f"❌ No checkpoint data found for {test_id}")
            continue
        
        methods_data = query_results.get('methods', {})
        
        # Dictionary to save results for this test (similar format to original dataset)
        test_results = {
            "map_name": map_name,
            "query_name": query_name,
            "query_file": query_file,
            "query": query,
            "methods": {}
        }
        
        # Copy A* result from original dataset (don't run again)
        if 'A*' in methods_data:
            test_results["methods"]["A*"] = methods_data['A*']
            print(f"📋 Copied A* baseline from dataset: ops={methods_data['A*'].get('operation')}, storage={methods_data['A*'].get('storage')}, length={methods_data['A*'].get('length')}")
        
        # Run all alpha values for this query
        for alpha_decay in alpha_values:
            print(f"\n   🎯 Testing alpha_decay = {alpha_decay}")
            
            # Test LLM-A* Adaptive with checkpoints
            llm_checkpoints_raw = methods_data.get('LLM-A*', {}).get('checkpoints')
            llm_checkpoints = convert_checkpoints_to_tuples(llm_checkpoints_raw)
            if llm_checkpoints:
                try:
                    llm_astar_adaptive = LLMAStar(llm='deepseek', prompt='repe', improved=False, 
                                                 adaptive=True, alpha_decay=alpha_decay)
                    llm_result = llm_astar_adaptive.searching_with_predefined_targets(
                        query, llm_checkpoints, f'temp_llm_adaptive_{alpha_decay}.png', no_plot=True)
                    method_name = f"LLM-A* (Adaptive α={alpha_decay})"
                    test_results["methods"][method_name] = extract_result_fields(llm_result)
                    # Only print the summary, not the full result dictionary
                    print(f"      ✅ LLM-A* α={alpha_decay}: ops={llm_result.get('operation')}, storage={llm_result.get('storage')}, length={llm_result.get('length'):.2f}")
                except Exception as e:
                    print(f"      ❌ LLM-A* α={alpha_decay} failed: {e}")
                    method_name = f"LLM-A* (Adaptive α={alpha_decay})"
                    test_results["methods"][method_name] = {"error": str(e)}
            else:
                print(f"      ❌ LLM-A* α={alpha_decay}: No checkpoints found")
                method_name = f"LLM-A* (Adaptive α={alpha_decay})"
                test_results["methods"][method_name] = {"error": "No LLM-A* checkpoints found"}
            
            # Test VLM-A* Adaptive using LLM framework with VLM checkpoints
            vlm_checkpoints_raw = methods_data.get('VLM-A*', {}).get('checkpoints')
            vlm_checkpoints = convert_checkpoints_to_tuples(vlm_checkpoints_raw)
            if vlm_checkpoints:
                try:
                    vlm_astar_adaptive = LLMAStar(llm='deepseek', prompt='repe', improved=False, 
                                                 adaptive=True, alpha_decay=alpha_decay)
                    vlm_result = vlm_astar_adaptive.searching_with_predefined_targets(
                        query, vlm_checkpoints, f'temp_vlm_adaptive_{alpha_decay}.png', no_plot=True)
                    method_name = f"VLM-A* (Adaptive α={alpha_decay})"
                    test_results["methods"][method_name] = extract_result_fields(vlm_result)
                    # Only print the summary, not the full result dictionary
                    print(f"      ✅ VLM-A* α={alpha_decay}: ops={vlm_result.get('operation')}, storage={vlm_result.get('storage')}, length={vlm_result.get('length'):.2f}")
                except Exception as e:
                    print(f"      ❌ VLM-A* α={alpha_decay} failed: {e}")
                    method_name = f"VLM-A* (Adaptive α={alpha_decay})"
                    test_results["methods"][method_name] = {"error": str(e)}
            else:
                print(f"      ❌ VLM-A* α={alpha_decay}: No checkpoints found")
                method_name = f"VLM-A* (Adaptive α={alpha_decay})"
                test_results["methods"][method_name] = {"error": "No VLM-A* checkpoints found"}
        
        # Save results for this test
        all_results["results"][test_id] = test_results
        print(f"\n✅ Completed {test_id} with all alpha values")
    
    # Save all results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"adaptive_analysis_results_{timestamp}.json"

    # with open(results_filename, 'w') as f:
    #     json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"🎉 ADAPTIVE ANALYSIS COMPLETED!")
    print(f"📁 Results saved to: {results_filename}")
    print(f"📊 Alpha values tested: {alpha_values}")
    print(f"📊 Total queries processed: {len(query_files)}")
    print(f"{'='*80}")
    
    return results_filename

def extract_result_fields(result):
    """Extract operation, storage, length, checkpoints fields from result"""
    if result is None:
        return {"operation": None, "storage": None, "length": None, "path_found": None, "checkpoints": None, "error": "Result is None"}
    
    # Handle checkpoints/waypoints
    checkpoints = None
    if 'llm_output' in result:
        # For LLM-A*, VLM-A* - get waypoints from llm_output
        checkpoints = result.get('llm_output', None)
    elif 'checkpoints' in result:
        # Fallback for checkpoints field if available
        checkpoints = result.get('checkpoints', None)
    # For pure A*, checkpoints will be None
    
    return {
        "operation": result.get('operation', None),
        "storage": result.get('storage', None), 
        "length": result.get('length', None),
        "path_found": result.get('path_found', None),
        "checkpoints": checkpoints
    }

def analyze_adaptive_results(json_file):
    """Analyze adaptive results across different alpha values with A* as baseline (100%)"""
    
    # Read JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print("📊 ADAPTIVE ANALYSIS SUMMARY (A* = 100% Baseline)")
    print("="*80)
    
    alpha_values = data.get('alpha_values_tested', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    results = data.get('results', {})
    
    # First, calculate A* baseline averages
    astar_ops = []
    astar_storage = []
    astar_length = []
    
    for query_id, query_data in results.items():
        methods = query_data.get('methods', {})
        astar_result = methods.get('A*', {})
        if astar_result and 'error' not in astar_result:
            if astar_result.get('operation'): astar_ops.append(astar_result['operation'])
            if astar_result.get('storage'): astar_storage.append(astar_result['storage'])
            if astar_result.get('length'): astar_length.append(astar_result['length'])
    
    # Calculate A* baseline averages
    astar_avg_ops = sum(astar_ops) / len(astar_ops) if astar_ops else 1
    astar_avg_storage = sum(astar_storage) / len(astar_storage) if astar_storage else 1
    astar_avg_length = sum(astar_length) / len(astar_length) if astar_length else 1
    
    print(f"\n📋 A* Baseline Averages:")
    print(f"   Operations: {astar_avg_ops:.1f}")
    print(f"   Storage: {astar_avg_storage:.1f}")
    print(f"   Length: {astar_avg_length:.2f}")
    
    # Create summary table with percentages
    print(f"\n{'Alpha':<8} {'LLM-A* Ops%':<12} {'LLM-A* Storage%':<15} {'LLM-A* Length%':<15} {'VLM-A* Ops%':<12} {'VLM-A* Storage%':<15} {'VLM-A* Length%':<15}")
    print("-" * 100)
    
    alpha_summary = {}
    
    for alpha in alpha_values:
        # Calculate averages for this alpha
        llm_ops = []
        llm_storage = []
        llm_length = []
        vlm_ops = []
        vlm_storage = []
        vlm_length = []
        
        for query_id, query_data in results.items():
            methods = query_data.get('methods', {})
            
            # LLM-A* Adaptive for this alpha
            llm_method_name = f"LLM-A* (Adaptive α={alpha})"
            llm_result = methods.get(llm_method_name, {})
            if llm_result and 'error' not in llm_result:
                if llm_result.get('operation'): llm_ops.append(llm_result['operation'])
                if llm_result.get('storage'): llm_storage.append(llm_result['storage'])
                if llm_result.get('length'): llm_length.append(llm_result['length'])
            
            # VLM-A* Adaptive for this alpha
            vlm_method_name = f"VLM-A* (Adaptive α={alpha})"
            vlm_result = methods.get(vlm_method_name, {})
            if vlm_result and 'error' not in vlm_result:
                if vlm_result.get('operation'): vlm_ops.append(vlm_result['operation'])
                if vlm_result.get('storage'): vlm_storage.append(vlm_result['storage'])
                if vlm_result.get('length'): vlm_length.append(vlm_result['length'])
        
        # Calculate averages
        llm_avg_ops = sum(llm_ops) / len(llm_ops) if llm_ops else 0
        llm_avg_storage = sum(llm_storage) / len(llm_storage) if llm_storage else 0
        llm_avg_length = sum(llm_length) / len(llm_length) if llm_length else 0
        
        vlm_avg_ops = sum(vlm_ops) / len(vlm_ops) if vlm_ops else 0
        vlm_avg_storage = sum(vlm_storage) / len(vlm_storage) if vlm_storage else 0
        vlm_avg_length = sum(vlm_length) / len(vlm_length) if vlm_length else 0
        
        # Calculate percentages relative to A* baseline
        llm_ops_pct = (llm_avg_ops / astar_avg_ops * 100) if astar_avg_ops > 0 and llm_avg_ops > 0 else 0
        llm_storage_pct = (llm_avg_storage / astar_avg_storage * 100) if astar_avg_storage > 0 and llm_avg_storage > 0 else 0
        llm_length_pct = (llm_avg_length / astar_avg_length * 100) if astar_avg_length > 0 and llm_avg_length > 0 else 0
        
        vlm_ops_pct = (vlm_avg_ops / astar_avg_ops * 100) if astar_avg_ops > 0 and vlm_avg_ops > 0 else 0
        vlm_storage_pct = (vlm_avg_storage / astar_avg_storage * 100) if astar_avg_storage > 0 and vlm_avg_storage > 0 else 0
        vlm_length_pct = (vlm_avg_length / astar_avg_length * 100) if astar_avg_length > 0 and vlm_avg_length > 0 else 0
        
        # Store for later analysis
        alpha_summary[float(alpha)] = {
            'llm': {'ops': llm_ops_pct, 'storage': llm_storage_pct, 'length': llm_length_pct},
            'vlm': {'ops': vlm_ops_pct, 'storage': vlm_storage_pct, 'length': vlm_length_pct}
        }
        
        # Print row with percentages
        print(f"{alpha:<8} {llm_ops_pct:<12.1f} {llm_storage_pct:<15.1f} {llm_length_pct:<15.1f} {vlm_ops_pct:<12.1f} {vlm_storage_pct:<15.1f} {vlm_length_pct:<15.1f}")
    
    # Calculate overall averages across all alpha values
    print(f"\n{'='*80}")
    print("📊 OVERALL AVERAGE PERFORMANCE (All Alpha Values)")
    print("="*80)
    
    # Calculate averages across all alpha values
    all_llm_ops_pct = []
    all_llm_storage_pct = []
    all_llm_length_pct = []
    all_vlm_ops_pct = []
    all_vlm_storage_pct = []
    all_vlm_length_pct = []
    
    for alpha in alpha_values:
        if alpha_summary[float(alpha)]['llm']['ops'] > 0:
            all_llm_ops_pct.append(alpha_summary[float(alpha)]['llm']['ops'])
        if alpha_summary[float(alpha)]['llm']['storage'] > 0:
            all_llm_storage_pct.append(alpha_summary[float(alpha)]['llm']['storage'])
        if alpha_summary[float(alpha)]['llm']['length'] > 0:
            all_llm_length_pct.append(alpha_summary[float(alpha)]['llm']['length'])
            
        if alpha_summary[float(alpha)]['vlm']['ops'] > 0:
            all_vlm_ops_pct.append(alpha_summary[float(alpha)]['vlm']['ops'])
        if alpha_summary[float(alpha)]['vlm']['storage'] > 0:
            all_vlm_storage_pct.append(alpha_summary[float(alpha)]['vlm']['storage'])
        if alpha_summary[float(alpha)]['vlm']['length'] > 0:
            all_vlm_length_pct.append(alpha_summary[float(alpha)]['vlm']['length'])
    
    # Calculate overall averages
    overall_llm_ops = sum(all_llm_ops_pct) / len(all_llm_ops_pct) if all_llm_ops_pct else 0
    overall_llm_storage = sum(all_llm_storage_pct) / len(all_llm_storage_pct) if all_llm_storage_pct else 0
    overall_llm_length = sum(all_llm_length_pct) / len(all_llm_length_pct) if all_llm_length_pct else 0
    
    overall_vlm_ops = sum(all_vlm_ops_pct) / len(all_vlm_ops_pct) if all_vlm_ops_pct else 0
    overall_vlm_storage = sum(all_vlm_storage_pct) / len(all_vlm_storage_pct) if all_vlm_storage_pct else 0
    overall_vlm_length = sum(all_vlm_length_pct) / len(all_vlm_length_pct) if all_vlm_length_pct else 0
    
    print(f"\n📊 LLM-A* (Adaptive) - Overall Average Performance:")
    print(f"   Operations: {overall_llm_ops:.1f}% (A* = 100%)")
    print(f"   Storage:    {overall_llm_storage:.1f}% (A* = 100%)")
    print(f"   Length:     {overall_llm_length:.1f}% (A* = 100%)")
    
    print(f"\n📊 VLM-A* (Adaptive) - Overall Average Performance:")
    print(f"   Operations: {overall_vlm_ops:.1f}% (A* = 100%)")
    print(f"   Storage:    {overall_vlm_storage:.1f}% (A* = 100%)")
    print(f"   Length:     {overall_vlm_length:.1f}% (A* = 100%)")
    
    # Find optimal alpha values
    print(f"\n{'='*80}")
    print("🏆 OPTIMAL ALPHA VALUES")
    print("="*80)
    
    # For LLM-A*
    best_llm_ops_alpha = min(alpha_summary.keys(), key=lambda a: alpha_summary[a]['llm']['ops'] if alpha_summary[a]['llm']['ops'] > 0 else float('inf'))
    best_llm_storage_alpha = min(alpha_summary.keys(), key=lambda a: alpha_summary[a]['llm']['storage'] if alpha_summary[a]['llm']['storage'] > 0 else float('inf'))
    best_llm_length_alpha = min(alpha_summary.keys(), key=lambda a: alpha_summary[a]['llm']['length'] if alpha_summary[a]['llm']['length'] > 0 else float('inf'))
    
    print(f"📊 LLM-A* (Adaptive):")
    print(f"   Best Operations: α = {best_llm_ops_alpha} (avg: {alpha_summary[best_llm_ops_alpha]['llm']['ops']:.1f})")
    print(f"   Best Storage:    α = {best_llm_storage_alpha} (avg: {alpha_summary[best_llm_storage_alpha]['llm']['storage']:.1f})")
    print(f"   Best Length:     α = {best_llm_length_alpha} (avg: {alpha_summary[best_llm_length_alpha]['llm']['length']:.2f})")
    
    # For VLM-A*
    best_vlm_ops_alpha = min(alpha_summary.keys(), key=lambda a: alpha_summary[a]['vlm']['ops'] if alpha_summary[a]['vlm']['ops'] > 0 else float('inf'))
    best_vlm_storage_alpha = min(alpha_summary.keys(), key=lambda a: alpha_summary[a]['vlm']['storage'] if alpha_summary[a]['vlm']['storage'] > 0 else float('inf'))
    best_vlm_length_alpha = min(alpha_summary.keys(), key=lambda a: alpha_summary[a]['vlm']['length'] if alpha_summary[a]['vlm']['length'] > 0 else float('inf'))
    
    print(f"\n📊 VLM-A* (Adaptive):")
    print(f"   Best Operations: α = {best_vlm_ops_alpha} (avg: {alpha_summary[best_vlm_ops_alpha]['vlm']['ops']:.1f})")
    print(f"   Best Storage:    α = {best_vlm_storage_alpha} (avg: {alpha_summary[best_vlm_storage_alpha]['vlm']['storage']:.1f})")
    print(f"   Best Length:     α = {best_vlm_length_alpha} (avg: {alpha_summary[best_vlm_length_alpha]['vlm']['length']:.2f})")
    
    # Save analysis results
    analysis_results = {
        'alpha_summary': alpha_summary,
        'best_alpha': {
            'llm': {
                'operations': best_llm_ops_alpha,
                'storage': best_llm_storage_alpha,
                'length': best_llm_length_alpha
            },
            'vlm': {
                'operations': best_vlm_ops_alpha,
                'storage': best_vlm_storage_alpha,
                'length': best_vlm_length_alpha
            }
        },
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    output_file = json_file.replace('.json', '_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Analysis results saved to: {output_file}")
    return analysis_results

def parse_log_to_json(log_file_path):
    """Parse log file and convert to JSON format similar to run_adaptive_analysis output"""
    
    print(f"\n{'='*80}")
    print(f"📄 PARSING LOG FILE: {log_file_path}")
    print("="*80)
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except Exception as e:
        print(f"❌ Error reading log file: {e}")
        return None
    
    # Split log into lines for processing
    lines = log_content.split('\n')
    
    # Initialize result structure
    parsed_results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source": log_file_path,
        "alpha_values_tested": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "note": "Parsed from log file - adaptive analysis results",
        "results": {}
    }
    
    # State variables for parsing
    current_query = None
    current_alpha = None
    current_query_results = None
    
    # Regex patterns for parsing
    import re
    
    query_pattern = r'📊 Processing \d+/\d+: (map_\d+_query_\d+)'
    baseline_pattern = r'📋 Copied A\* baseline from dataset: ops=(\d+), storage=(\d+), length=([\d.]+)'
    alpha_pattern = r'🎯 Testing alpha_decay = ([\d.]+)'
    llm_result_pattern = r'✅ LLM-A\* α=([\d.]+): ops=(\d+), storage=(\d+), length=([\d.]+)'
    vlm_result_pattern = r'✅ VLM-A\* α=([\d.]+): ops=(\d+), storage=(\d+), length=([\d.]+)'
    completed_pattern = r'✅ Completed (map_\d+_query_\d+) with all alpha values'
    
    print(f"🔍 Parsing {len(lines)} lines from log file...")
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Check for query start
        query_match = re.search(query_pattern, line)
        if query_match:
            current_query = query_match.group(1)
            current_query_results = {
                "map_name": current_query.split('_')[0] + '_' + current_query.split('_')[1], 
                "query_name": current_query.split('_')[2] + '_' + current_query.split('_')[3],
                "methods": {}
            }
            print(f"   🆔 Found query: {current_query}")
            continue
        
        # Check for A* baseline
        if current_query:
            baseline_match = re.search(baseline_pattern, line)
            if baseline_match:
                ops, storage, length = baseline_match.groups()
                current_query_results["methods"]["A*"] = {
                    "operation": int(ops),
                    "storage": int(storage),
                    "length": float(length),
                    "path_found": True
                }
                print(f"      📋 A* baseline: ops={ops}, storage={storage}, length={length}")
                continue
            
            # Check for LLM-A* results
            llm_match = re.search(llm_result_pattern, line)
            if llm_match:
                alpha, ops, storage, length = llm_match.groups()
                method_name = f"LLM-A* (Adaptive α={alpha})"
                current_query_results["methods"][method_name] = {
                    "operation": int(ops),
                    "storage": int(storage), 
                    "length": float(length),
                    "path_found": True,
                    "checkpoints": None  # Not available in log
                }
                print(f"      ✅ LLM-A* α={alpha}: ops={ops}, storage={storage}, length={length}")
                continue
            
            # Check for VLM-A* results  
            vlm_match = re.search(vlm_result_pattern, line)
            if vlm_match:
                alpha, ops, storage, length = vlm_match.groups()
                method_name = f"VLM-A* (Adaptive α={alpha})"
                current_query_results["methods"][method_name] = {
                    "operation": int(ops),
                    "storage": int(storage),
                    "length": float(length), 
                    "path_found": True,
                    "checkpoints": None  # Not available in log
                }
                print(f"      ✅ VLM-A* α={alpha}: ops={ops}, storage={storage}, length={length}")
                continue
            
            # Check for query completion
            completed_match = re.search(completed_pattern, line)
            if completed_match and current_query:
                completed_query = completed_match.group(1)
                if completed_query == current_query:
                    # Save the completed query results
                    parsed_results["results"][current_query] = current_query_results
                    print(f"   ✅ Completed parsing: {current_query}")
                    current_query = None
                    current_query_results = None
                continue
    
    # Update total queries
    parsed_results["total_queries"] = len(parsed_results["results"])
    
    # Save parsed results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_base_name = os.path.splitext(os.path.basename(log_file_path))[0]  # Remove extension
    output_filename = f"parsed_{log_base_name}_{timestamp}.json"
    
    with open(output_filename, 'w') as f:
        json.dump(parsed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"🎉 LOG PARSING COMPLETED!")
    print(f"📁 Parsed results saved to: {output_filename}")
    print(f"📊 Total queries parsed: {parsed_results['total_queries']}")
    print(f"{'='*80}")
    
    return output_filename

def analyze_log_results(log_file_path):
    """Parse log file and analyze adaptive results - combines parsing and analysis"""
    
    # First parse the log file
    json_filename = parse_log_to_json(log_file_path)
    if not json_filename:
        print("❌ Failed to parse log file")
        return None
    
    # Then analyze the parsed results using existing function
    analysis_results = analyze_adaptive_results(json_filename)
    
    # Create custom analysis filename based on original log filename
    log_base_name = os.path.splitext(os.path.basename(log_file_path))[0]  # Remove .log extension
    custom_analysis_filename = f"alpha_log/{log_base_name}_analysis.json"
    
    # Save analysis with custom filename
    if analysis_results:
        with open(custom_analysis_filename, 'w') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Custom analysis results saved to: {custom_analysis_filename}")
    
    return analysis_results

def process_all_alpha_log_files():
    """Process all log files in alpha_log directory and generate analysis JSON files"""
    
    print(f"\n{'#'*80}")
    print("🚀 PROCESSING ALL ALPHA LOG FILES")
    print(f"{'#'*80}")
    
    # Get all log files in alpha_log directory
    alpha_log_dir = "alpha_log"
    if not os.path.exists(alpha_log_dir):
        print(f"❌ Directory not found: {alpha_log_dir}")
        return
    
    # Find all .log files
    log_files = []
    for file in os.listdir(alpha_log_dir):
        if file.endswith('.log'):
            log_file_path = os.path.join(alpha_log_dir, file)
            log_files.append(log_file_path)
    
    log_files.sort()  # Sort alphabetically
    
    print(f"📁 Found {len(log_files)} log files to process:")
    for i, log_file in enumerate(log_files, 1):
        print(f"   {i}. {log_file}")
    
    # Process each log file
    processed_count = 0
    for log_file_path in log_files:
        print(f"\n{'='*80}")
        print(f"📄 Processing: {log_file_path}")
        print(f"{'='*80}")
        
        try:
            # Extract base filename for analysis output
            log_base_name = os.path.splitext(os.path.basename(log_file_path))[0]
            analysis_filename = f"alpha_analyst/{log_base_name}_analysis.json"
            
            # Always process and overwrite existing files
            print(f"🔄 Processing and will overwrite: {analysis_filename}")
            
            # Parse log file and create analysis
            analysis_results = process_single_log_file(log_file_path)
            
            if analysis_results:
                # Ensure alpha_analyst directory exists
                os.makedirs("alpha_analyst", exist_ok=True)
                
                # Save analysis results (overwrite if exists)
                with open(analysis_filename, 'w') as f:
                    json.dump(analysis_results, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Analysis saved/overwritten: {analysis_filename}")
                processed_count += 1
            else:
                print(f"❌ Failed to process: {log_file_path}")
                
        except Exception as e:
            print(f"❌ Error processing {log_file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'#'*80}")
    print(f"🎉 PROCESSING COMPLETED!")
    print(f"📊 Successfully processed: {processed_count}/{len(log_files)} files")
    print(f"📁 Analysis files saved in: alpha_analyst/")
    print(f"{'#'*80}")

def process_single_log_file(log_file_path):
    """Process a single log file and return analysis results in the desired format"""
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except Exception as e:
        print(f"❌ Error reading log file: {e}")
        return None
    
    # Parse log content to extract alpha analysis data
    alpha_results = parse_alpha_results_from_log(log_content)
    
    if not alpha_results:
        print(f"❌ No alpha results found in log file")
        return None
    
    # Calculate summary statistics for each alpha value
    alpha_summary = calculate_alpha_summary(alpha_results)
    
    # Create analysis results in the desired format
    analysis_results = {
        "alpha_summary": alpha_summary,
        "source_log": log_file_path,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "note": "Analysis generated from alpha log file, using A* as baseline (100%)"
    }
    
    return analysis_results

def parse_alpha_results_from_log(log_content):
    """Parse alpha analysis results from log content"""
    
    import re
    lines = log_content.split('\n')
    
    # Storage for parsed results
    alpha_results = {}  # {alpha: {query: {llm: {...}, vlm: {...}, astar: {...}}}}
    
    # State variables
    current_query = None
    current_alpha = None
    astar_baseline = None
    
    # Regex patterns
    query_pattern = r'📊 Processing \d+/\d+: (map_\d+_query_\d+)'
    baseline_pattern = r'📋 Copied A\* baseline from dataset: ops=(\d+), storage=(\d+), length=([\d.]+)'
    alpha_pattern = r'🎯 Testing alpha_decay = ([\d.]+)'
    llm_result_pattern = r'✅ LLM-A\* α=([\d.]+): ops=(\d+), storage=(\d+), length=([\d.]+)'
    vlm_result_pattern = r'✅ VLM-A\* α=([\d.]+): ops=(\d+), storage=(\d+), length=([\d.]+)'
    
    print(f"🔍 Parsing alpha results from {len(lines)} lines...")
    
    for line in lines:
        line = line.strip()
        
        # Check for query start
        query_match = re.search(query_pattern, line)
        if query_match:
            current_query = query_match.group(1)
            continue
        
        # Check for A* baseline
        baseline_match = re.search(baseline_pattern, line)
        if baseline_match:
            astar_baseline = {
                'ops': int(baseline_match.group(1)),
                'storage': int(baseline_match.group(2)),
                'length': float(baseline_match.group(3))
            }
            continue
        
        # Check for alpha value
        alpha_match = re.search(alpha_pattern, line)
        if alpha_match:
            current_alpha = alpha_match.group(1)
            
            # Initialize alpha results if not exists
            if current_alpha not in alpha_results:
                alpha_results[current_alpha] = {}
            
            # Initialize query results if not exists
            if current_query not in alpha_results[current_alpha]:
                alpha_results[current_alpha][current_query] = {
                    'astar': astar_baseline.copy() if astar_baseline else None
                }
            continue
        
        # Check for LLM results
        llm_match = re.search(llm_result_pattern, line)
        if llm_match and current_alpha and current_query:
            alpha_results[current_alpha][current_query]['llm'] = {
                'ops': int(llm_match.group(2)),
                'storage': int(llm_match.group(3)),
                'length': float(llm_match.group(4))
            }
            continue
        
        # Check for VLM results
        vlm_match = re.search(vlm_result_pattern, line)
        if vlm_match and current_alpha and current_query:
            alpha_results[current_alpha][current_query]['vlm'] = {
                'ops': int(vlm_match.group(2)),
                'storage': int(vlm_match.group(3)),
                'length': float(vlm_match.group(4))
            }
            continue
    
    print(f"✅ Found results for {len(alpha_results)} alpha values")
    for alpha, queries in alpha_results.items():
        print(f"   α={alpha}: {len(queries)} queries")
    
    return alpha_results

def calculate_alpha_summary(alpha_results):
    """Calculate summary statistics for each alpha value using A* as baseline (100%)"""
    
    alpha_summary = {}
    
    for alpha, queries in alpha_results.items():
        print(f"\n📊 Calculating summary for α={alpha} ({len(queries)} queries)")
        
        # Collect all metrics for this alpha
        llm_ops_ratios = []
        llm_storage_ratios = []
        llm_length_ratios = []
        
        vlm_ops_ratios = []
        vlm_storage_ratios = []
        vlm_length_ratios = []
        
        valid_queries = 0
        
        for query_name, results in queries.items():
            astar = results.get('astar')
            llm = results.get('llm')
            vlm = results.get('vlm')
            
            # Skip if missing essential data
            if not astar or not llm or not vlm:
                continue
            
            valid_queries += 1
            
            # Calculate ratios as percentage of A* (A* = 100%)
            if astar['ops'] > 0:
                llm_ops_ratios.append((llm['ops'] / astar['ops']) * 100)
                vlm_ops_ratios.append((vlm['ops'] / astar['ops']) * 100)
            
            if astar['storage'] > 0:
                llm_storage_ratios.append((llm['storage'] / astar['storage']) * 100)
                vlm_storage_ratios.append((vlm['storage'] / astar['storage']) * 100)
            
            if astar['length'] > 0:
                llm_length_ratios.append((llm['length'] / astar['length']) * 100)
                vlm_length_ratios.append((vlm['length'] / astar['length']) * 100)
        
        # Calculate averages
        if valid_queries > 0:
            alpha_summary[alpha] = {
                "llm": {
                    "ops": np.mean(llm_ops_ratios) if llm_ops_ratios else 0,
                    "storage": np.mean(llm_storage_ratios) if llm_storage_ratios else 0,
                    "length": np.mean(llm_length_ratios) if llm_length_ratios else 0
                },
                "vlm": {
                    "ops": np.mean(vlm_ops_ratios) if vlm_ops_ratios else 0,
                    "storage": np.mean(vlm_storage_ratios) if vlm_storage_ratios else 0,
                    "length": np.mean(vlm_length_ratios) if vlm_length_ratios else 0
                }
            }
            
            print(f"   Valid queries: {valid_queries}")
            print(f"   LLM avg: ops={alpha_summary[alpha]['llm']['ops']:.2f}%, storage={alpha_summary[alpha]['llm']['storage']:.2f}%, length={alpha_summary[alpha]['llm']['length']:.2f}%")
            print(f"   VLM avg: ops={alpha_summary[alpha]['vlm']['ops']:.2f}%, storage={alpha_summary[alpha]['vlm']['storage']:.2f}%, length={alpha_summary[alpha]['vlm']['length']:.2f}%")
        else:
            print(f"   ❌ No valid queries found for α={alpha}")
    
    return alpha_summary

def get_checkpoint_files_by_option(option, checkpoint_file_list):
    """Get subset of checkpoint files based on option parameter"""
    if option == 1:
        return checkpoint_file_list[:2]  # First 2 files
    elif option == 2:
        return checkpoint_file_list[2:4]  # Next 2 files (index 2-3)
    elif option == 3:
        return checkpoint_file_list[4:6]  # Next 2 files (index 4-5)
    elif option == 4:
        return checkpoint_file_list[6:8]  # Next 2 files (index 6-7)
    elif option == 5:
        return checkpoint_file_list[8:10]  # Next 2 files (index 8-9)
    elif option == 6:
        return checkpoint_file_list[10:11]  # Last 2 files (index 10-11)
    elif option == 7:
        return checkpoint_file_list[11:]
    else:
        print(f"❌ Invalid option: {option}. Valid options are 1-7.")
        return []

if __name__ == "__main__":
    # # Parse command line arguments
    # parser = argparse.ArgumentParser(description='Run adaptive analysis with different checkpoint file subsets')
    # parser.add_argument('--option', type=int, choices=[1, 2, 3, 4, 5, 6, 7], required=True,
    #                     help='Option to select checkpoint files subset: 1=first 2 files, 2=next 2 files, etc.')
    # args = parser.parse_args()
    
    # # Configuration
    # checkpoint_file_list = [
    #     "Ly/dataset_results_deepseek_llama-4-Marverick_0.9_20250919_144105.json",
    #     "Ly/dataset_results_gpt_llama-4-Marverick_0.9_20250919_184150.json",
    #     "Ly/dataset_results_llama3_fpt_llama-4-Marverick_0.9_20250919_141054.json",
    #     "Ly/dataset_results_qwen_llama-4-Marverick_0.9_20250919_184220.json",

    #     "Ly/dataset_results_deepseek_gemma_20250925_151814.json",
    #     "Ly/dataset_results_gpt_gemma_20250925_154501.json",
    #     "Ly/dataset_results_llama3_gemma_20250925_153222.json",
    #     "Ly/dataset_results_llama3_qwen_20250925_153729.json",

    #     "Ly/dataset_results_deepseek_qwen_20250925_152553.json",
    #     "Ly/dataset_results_gpt_qwen_20250925_154646.json",
    #     "Ly/dataset_results_qwen_qwen_20250925_154243.json",
    #     "Ly/dataset_results_qwen_gemma_20250925_154102.json"
    # ]
    
    # # Get subset of files based on option
    # selected_files = get_checkpoint_files_by_option(args.option, checkpoint_file_list)
    
    # # Parse command line arguments
    # parser = argparse.ArgumentParser(description='Run adaptive analysis with different checkpoint file subsets')
    # parser.add_argument('--option', type=int, choices=[1, 2, 3, 4, 5, 6, 7], required=True,
    #                     help='Option to select checkpoint files subset: 1=first 2 files, 2=next 2 files, etc.')
    # args = parser.parse_args()
    
    # # Configuration
    # checkpoint_file_list = [
    #     "Ly/dataset_results_deepseek_llama-4-Marverick_0.9_20250919_144105.json",
    #     "Ly/dataset_results_gpt_llama-4-Marverick_0.9_20250919_184150.json",
    #     "Ly/dataset_results_llama3_fpt_llama-4-Marverick_0.9_20250919_141054.json",
    #     "Ly/dataset_results_qwen_llama-4-Marverick_0.9_20250919_184220.json",

    #     "Ly/dataset_results_deepseek_gemma_20250925_151814.json",
    #     "Ly/dataset_results_gpt_gemma_20250925_154501.json",
    #     "Ly/dataset_results_llama3_gemma_20250925_153222.json",
    #     "Ly/dataset_results_llama3_qwen_20250925_153729.json",

    #     "Ly/dataset_results_deepseek_qwen_20250925_152553.json",
    #     "Ly/dataset_results_gpt_qwen_20250925_154646.json",
    #     "Ly/dataset_results_qwen_qwen_20250925_154243.json",
    #     "Ly/dataset_results_qwen_gemma_20250925_154102.json"
    # ]
    
    # # Get subset of files based on option
    # selected_files = get_checkpoint_files_by_option(args.option, checkpoint_file_list)
    
    # print(f"\n{'#'*80}")
    # print(f"🎯 Option {args.option} selected - Processing {len(selected_files)} files:")
    # for i, file in enumerate(selected_files, 1):
    #     print(f"   {i}. {file}")
    # print(f"{'#'*80}")
    
    # for checkpoint_file in selected_files:
    #     alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #     print(f"\n{'#'*80}\n")
    #     print(checkpoint_file)
    #     print(f"🚀 Starting Adaptive Analysis")
    #     print(f"📁 Checkpoint source: {checkpoint_file}")
    #     print(f"🎯 Alpha values to test: {alpha_values}")
        
    #     # Run adaptive analysis
    #     results_filename = run_adaptive_analysis(checkpoint_file, alpha_values)

    # NEW: Process all alpha log files to generate analysis JSON files
    process_all_alpha_log_files()
    
    # # OLD: Test parsing and analyzing specific log files 
    # alpha_log_files = [
    #     "alpha_log/varied_decay_qwen_gemma.log",
    #     "alpha_log/varied_decay_qwen_llama4.log", 
    #     "alpha_log/varied_decay_qwen_qwen.log",
    # ]
    
    # print(f"\n{'#'*80}")
    # print("🧪 ANALYZING SPECIFIC LOG FILES")
    # print(f"{'#'*80}")
    
    # for log_file_path in alpha_log_files:
    #     if os.path.exists(log_file_path):
    #         print(f"\n{'='*60}")
    #         print(f"📄 Processing: {log_file_path}")
    #         print(f"{'='*60}")
    #         analyze_log_results(log_file_path)
    #     else:
    #         print(f"❌ Log file not found: {log_file_path}")
    
    # print(f"\n{'#'*80}")
    # print("🎉 ALL LOG FILE ANALYSIS COMPLETED!")
    # print(f"{'#'*80}")
    
    # if results_filename:
    #     # Analyze results
    #     analyze_adaptive_results(results_filename)
    # else:
    #     print("❌ Failed to run adaptive analysis")