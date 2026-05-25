#!/usr/bin/env python3
"""
Script để tạo visualizations cho tất cả các đường đi LLM-A* từ file JSON kết quả.
Tạo 200 ảnh bản đồ với đường đi nối checkpoint của phương pháp LLM-A*.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
import os
from pathlib import Path


def draw_map_with_path(query_data, method_data, output_path):
    """
    Vẽ bản đồ với barriers, start/goal points và đường đi LLM-A*
    
    Args:
        query_data: Dictionary chứa thông tin về map (size, barriers, start, goal)
        method_data: Dictionary chứa thông tin về phương pháp LLM-A* (checkpoints)
        output_path: Đường dẫn file output
    """
    # Lấy thông tin bản đồ
    # Kiểm tra xem có size không, nếu không thì sử dụng range_x và range_y
    if 'size' in query_data:
        map_size = query_data['size']  # [width, height]
    elif 'range_x' in query_data and 'range_y' in query_data:
        map_size = [query_data['range_x'][1], query_data['range_y'][1]]
    else:
        print(f"Không tìm thấy thông tin kích thước map cho {output_path}")
        return
    horizontal_barriers = query_data.get('horizontal_barriers', [])
    vertical_barriers = query_data.get('vertical_barriers', [])
    start_point = query_data.get('start')
    goal_point = query_data.get('goal')
    
    # Kiểm tra các field bắt buộc
    if not start_point or not goal_point:
        print(f"Thiếu thông tin start/goal point cho {output_path}")
        return
    
    # Lấy checkpoints từ LLM-A*
    checkpoints = method_data.get('checkpoints', [])
    
    if not checkpoints:
        print(f"Không có checkpoints cho {output_path}")
        return
    
    # Tạo figure và axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Thiết lập giới hạn trục
    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    ax.set_aspect('equal')
    
    # Vẽ horizontal barriers
    for barrier in horizontal_barriers:
        y, x_start, x_end = barrier
        # Vẽ barrier như một đường thẳng ngang
        ax.plot([x_start, x_end], [y, y], 'k-', linewidth=3, solid_capstyle='butt')
    
    # Vẽ vertical barriers  
    for barrier in vertical_barriers:
        x, y_start, y_end = barrier
        # Vẽ barrier như một đường thẳng dọc
        ax.plot([x, x], [y_start, y_end], 'k-', linewidth=3, solid_capstyle='butt')
    
    # Vẽ start point (màu xanh lá)
    ax.plot(start_point[0], start_point[1], 'go', markersize=12, label='Start', markeredgecolor='black', markeredgewidth=2)
    
    # Vẽ goal point (màu đỏ)
    ax.plot(goal_point[0], goal_point[1], 'rs', markersize=12, label='Goal', markeredgecolor='black', markeredgewidth=2)
    
    # Vẽ đường đi qua các checkpoints
    if len(checkpoints) >= 2:
        # Tách x và y coordinates
        path_x = [point[0] for point in checkpoints]
        path_y = [point[1] for point in checkpoints]
        
        # Vẽ đường đi (màu xanh dương)
        ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.8, label='LLM-A* Path')
        
        # Vẽ các checkpoints (màu cam)
        for i, checkpoint in enumerate(checkpoints):
            if i == 0 or i == len(checkpoints) - 1:
                continue  # Bỏ qua start và goal vì đã vẽ rồi
            ax.plot(checkpoint[0], checkpoint[1], 'o', color='orange', markersize=6, 
                   markeredgecolor='black', markeredgewidth=1)
    
    # Thiết lập nhãn và tiêu đề
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title(f'Map Visualization with LLM-A* Path\n{os.path.basename(output_path)[:-4]}', fontsize=14, fontweight='bold')
    
    # Thêm legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Thêm grid
    ax.grid(True, alpha=0.3)
    
    # Lưu ảnh
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Đã lưu: {output_path}")


def process_all_queries(json_file_path, output_dir):
    """
    Xử lý tất cả các queries từ file JSON và tạo ảnh tương ứng
    
    Args:
        json_file_path: Đường dẫn đến file JSON kết quả
        output_dir: Thư mục output để lưu ảnh
    """
    # Đọc file JSON
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    total_queries = len(results)
    processed = 0
    skipped = 0
    
    print(f"Bắt đầu xử lý {total_queries} queries...")
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Duyệt qua tất cả các kết quả
    for query_key, query_result in results.items():
        try:
            # Lấy thông tin query và method data
            query_data = query_result['query']
            methods = query_result['methods']
            
            # Kiểm tra xem có VLM-A* không
            if 'VLM-A*' not in methods:
                print(f"Không tìm thấy VLM-A* method cho {query_key}")
                skipped += 1
                continue
            
            llm_astar_data = methods['VLM-A*']
            
            # Kiểm tra xem có tìm thấy path không
            if not llm_astar_data.get('path_found', False):
                print(f"LLM-A* không tìm thấy path cho {query_key}")
                skipped += 1
                continue
            
            # Tạo tên file output
            output_filename = f"{query_key}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Vẽ map và lưu ảnh
            draw_map_with_path(query_data, llm_astar_data, output_path)
            processed += 1
            
        except Exception as e:
            print(f"Lỗi khi xử lý {query_key}: {str(e)}")
            skipped += 1
            continue
    
    print(f"\nHoàn thành!")
    print(f"Đã xử lý: {processed} ảnh")
    print(f"Bỏ qua: {skipped} queries")
    print(f"Tổng cộng: {total_queries} queries")


def main():
    """Hàm main"""
    # Đường dẫn đến file JSON kết quả
    current_dir = Path(__file__).parent.parent
    json_file = current_dir / "Ly/dataset_results_qwen_qwen_20250925_154243.json"
    
    # Thư mục output
    output_dir = current_dir / "Map"
    
    # Kiểm tra xem file JSON có tồn tại không
    if not json_file.exists():
        print(f"Không tìm thấy file JSON: {json_file}")
        return
    
    print(f"Đọc file: {json_file}")
    print(f"Output directory: {output_dir}")
    
    # Xử lý tất cả các queries
    process_all_queries(str(json_file), str(output_dir))


if __name__ == "__main__":
    main()