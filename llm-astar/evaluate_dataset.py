import os
import json
from datetime import datetime

# Tắt hoàn toàn matplotlib GUI trước khi import
import matplotlib
# matplotlib.use('Agg')  # Sử dụng backend không GUI
import matplotlib.pyplot as plt
# plt.ioff()  # Tắt interactive mode

from llmastar.pather import AStar, LLMVLMRunner

import openai
import os
from dotenv import load_dotenv

# Clear any existing environment variables
if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Get API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
if not api_key or api_key == 'your_new_openai_api_key_here':
    print("❌ Lỗi: API key không được tìm thấy hoặc chưa được cập nhật trong file .env")
    print(f"📁 Đường dẫn file .env: {env_path}")
    print("🔧 Vui lòng cập nhật OPENAI_API_KEY trong file .env với API key mới của bạn")
    exit(1)

openai.api_key = api_key
print(f"✅ API key loaded successfully: {api_key[:10]}...{api_key[-4:]}")

def run_dataset_evaluation():
    """Chạy evaluation trên toàn bộ dataset và lưu kết quả"""
    
    dataset_dir = "Dataset"
    if not os.path.exists(dataset_dir):
        print("Dataset folder not found!")
        return
    
    # Khởi tạo kết quả tổng hợp
    all_results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_queries": 0,
        "results": {}
    }
    
    # Lặp qua tất cả map folders
    map_folders = [f for f in os.listdir(dataset_dir) if f.startswith("map_")]
    map_folders.sort()
    
    for map_folder in map_folders:
        map_path = os.path.join(dataset_dir, map_folder)
        if not os.path.isdir(map_path):
            continue
            
        print(f"\n=== Processing {map_folder} ===")
        
        # Tìm tất cả file query JSON
        query_files = [f for f in os.listdir(map_path) if f.startswith("query_") and f.endswith(".json")]
        query_files.sort()
        
        map_results = {}
        
        for query_file in query_files:
            query_path = os.path.join(map_path, query_file)
            query_name = query_file.replace('.json', '')
            
            print(f"  Processing {query_name}...")
            
            try:
                # Đọc query từ file JSON
                with open(query_path, 'r') as f:
                    query = json.load(f)
                
                # Tạo thư mục output cho query này
                output_dir = os.path.join(map_path, f"{query_name}_results")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Chạy các thuật toán
                query_results = {}
                
                # 1. Chạy A*
                print(f"    Running A*...")
                astar_filepath = os.path.join(output_dir, "astar.png")
                astar_result = AStar().searching(query=query, filepath=astar_filepath)
                query_results['A*'] = astar_result
                
                # 2. Chạy LLM-A* và VLM-A* song song
                print(f"    Running LLM-A* and VLM-A*...")
                runner = LLMVLMRunner(llm='gpt', prompt='standard')
                llm_filepath = os.path.join(output_dir, "llm_standard.png")
                vlm_filepath = os.path.join(output_dir, "vlm_standard.png")
                
                llm_result, vlm_result = runner.run_both(query, llm_filepath, vlm_filepath)
                query_results['LLM-A*'] = llm_result
                query_results['VLM-A*'] = vlm_result
                
                # Lưu query info
                query_results['query'] = query
                query_results['query_file'] = query_file
                query_results['output_directory'] = output_dir
                
                map_results[query_name] = query_results
                all_results["total_queries"] += 1
                
                print(f"    ✅ Completed {query_name}")
                
            except Exception as e:
                print(f"    ❌ Error processing {query_name}: {e}")
                map_results[query_name] = {
                    "error": str(e),
                    "query_file": query_file
                }
        
        all_results["results"][map_folder] = map_results
    
    # Lưu kết quả tổng hợp
    results_filename = f"dataset_evaluation_results_{all_results['timestamp']}.json"
    with open(results_filename, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # In tóm tắt kết quả
    print_summary(all_results, results_filename)
    
    return all_results, results_filename

def print_summary(results, results_filename):
    """In tóm tắt kết quả evaluation"""
    
    print(f"\n{'='*60}")
    print(f"DATASET EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries processed: {results['total_queries']}")
    print(f"Results saved to: {results_filename}")
    
    # Thống kê theo từng map
    for map_name, map_results in results["results"].items():
        print(f"\n{map_name}:")
        
        successful_queries = 0
        failed_queries = 0
        
        for query_name, query_result in map_results.items():
            if "error" in query_result:
                failed_queries += 1
                print(f"  ❌ {query_name}: {query_result['error']}")
            else:
                successful_queries += 1
                
                # Tóm tắt kết quả cho query này
                print(f"  ✅ {query_name}:")
                for method in ['A*', 'LLM-A*', 'VLM-A*']:
                    if method in query_result:
                        result = query_result[method]
                        ops = result.get('operation', 'N/A')
                        length = result.get('length', 'N/A')
                        path_found = result.get('path_found', False)
                        
                        if isinstance(length, (int, float)):
                            length_str = f"{length:.2f}"
                        else:
                            length_str = str(length)
                        
                        status = "✓" if path_found else "✗"
                        print(f"    {method}: Ops={ops}, Length={length_str}, Path={status}")
        
        print(f"  Summary: {successful_queries} successful, {failed_queries} failed")
    
    # Thống kê tổng quát
    total_successful = sum(len([q for q in map_results.values() if "error" not in q]) 
                          for map_results in results["results"].values())
    total_failed = results["total_queries"] - total_successful
    
    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY:")
    print(f"  Successful: {total_successful}/{results['total_queries']} queries")
    print(f"  Failed: {total_failed}/{results['total_queries']} queries")
    print(f"  Success rate: {(total_successful/results['total_queries']*100):.1f}%")
    print(f"{'='*60}")

def analyze_performance(results_filename):
    """Phân tích hiệu năng của các thuật toán"""
    
    with open(results_filename, 'r') as f:
        results = json.load(f)
    
    methods = ['A*', 'LLM-A*', 'VLM-A*']
    performance_stats = {method: {'operations': [], 'lengths': [], 'success_count': 0, 'total_count': 0} 
                        for method in methods}
    
    # Thu thập dữ liệu
    for map_results in results["results"].values():
        for query_result in map_results.values():
            if "error" in query_result:
                continue
                
            for method in methods:
                if method in query_result:
                    result = query_result[method]
                    performance_stats[method]['total_count'] += 1
                    
                    if result.get('path_found', False):
                        performance_stats[method]['success_count'] += 1
                        performance_stats[method]['operations'].append(result.get('operation', 0))
                        if isinstance(result.get('length'), (int, float)):
                            performance_stats[method]['lengths'].append(result.get('length'))
    
    # In thống kê hiệu năng
    print(f"\n{'='*60}")
    print(f"PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    
    for method in methods:
        stats = performance_stats[method]
        print(f"\n{method}:")
        print(f"  Success rate: {stats['success_count']}/{stats['total_count']} ({stats['success_count']/max(stats['total_count'],1)*100:.1f}%)")
        
        if stats['operations']:
            avg_ops = sum(stats['operations']) / len(stats['operations'])
            min_ops = min(stats['operations'])
            max_ops = max(stats['operations'])
            print(f"  Operations: avg={avg_ops:.1f}, min={min_ops}, max={max_ops}")
        
        if stats['lengths']:
            avg_length = sum(stats['lengths']) / len(stats['lengths'])
            min_length = min(stats['lengths'])
            max_length = max(stats['lengths'])
            print(f"  Path length: avg={avg_length:.2f}, min={min_length:.2f}, max={max_length:.2f}")

if __name__ == "__main__":
    print("Starting dataset evaluation...")
    
    try:
        results, results_filename = run_dataset_evaluation()
        
        # Phân tích hiệu năng
        analyze_performance(results_filename)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📄 Detailed results: {results_filename}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
