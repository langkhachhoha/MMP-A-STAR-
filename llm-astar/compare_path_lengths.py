#!/usr/bin/env python3
"""
Script để tính toán và so sánh độ dài đường đi của LLM-A* so với A* 
cho các map hợp lệ trong file valid_path.txt.
Tính toán độ dài thực tế bằng cách tính khoảng cách Euclidean giữa các checkpoint.
"""

import json
import os
import math
from pathlib import Path


def calculate_euclidean_distance(point1, point2):
    """
    Tính khoảng cách Euclidean giữa hai điểm
    
    Args:
        point1: [x1, y1]
        point2: [x2, y2]
    
    Returns:
        float: Khoảng cách Euclidean
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return math.sqrt(dx * dx + dy * dy)


def calculate_path_length_from_checkpoints(checkpoints):
    """
    Tính độ dài đường đi từ danh sách các checkpoint
    
    Args:
        checkpoints: List các điểm [[x1, y1], [x2, y2], ...]
    
    Returns:
        float: Tổng độ dài đường đi
    """
    if not checkpoints or len(checkpoints) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(checkpoints) - 1):
        distance = calculate_euclidean_distance(checkpoints[i], checkpoints[i + 1])
        total_length += distance
    
    return total_length


def extract_map_query_from_filename(filename):
    """
    Trích xuất map_name và query_name từ tên file
    
    Args:
        filename: Tên file dạng "Map/map_2_query_13.png"
    
    Returns:
        tuple: (map_name, query_name) ví dụ ("map_2", "query_13")
    """
    # Bỏ đường dẫn "Map/" và đuôi ".png"
    basename = os.path.basename(filename).replace('.png', '')
    
    # Tách thành map_X_query_Y
    parts = basename.split('_')
    if len(parts) >= 4:  # map_X_query_Y
        map_name = f"{parts[0]}_{parts[1]}"  # map_X
        query_name = f"{parts[2]}_{parts[3]}"  # query_Y
        return map_name, query_name
    
    return None, None


def load_valid_paths(valid_path_file):
    """
    Đọc danh sách các map hợp lệ từ file valid_path.txt
    
    Args:
        valid_path_file: Đường dẫn đến file valid_path.txt
    
    Returns:
        list: Danh sách các (map_name, query_name)
    """
    valid_maps = []
    
    with open(valid_path_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                map_name, query_name = extract_map_query_from_filename(line)
                if map_name and query_name:
                    valid_maps.append((map_name, query_name))
    
    return valid_maps


def calculate_path_length_comparison(json_file_path, valid_path_file):
    """
    Tính toán so sánh độ dài đường đi giữa LLM-A* và A*
    
    Args:
        json_file_path: Đường dẫn đến file JSON kết quả
        valid_path_file: Đường dẫn đến file valid_path.txt
    
    Returns:
        dict: Kết quả phân tích
    """
    # Đọc danh sách các map hợp lệ
    valid_maps = load_valid_paths(valid_path_file)
    print(f"Tìm thấy {len(valid_maps)} map hợp lệ:")
    for map_name, query_name in valid_maps:
        print(f"  - {map_name}_{query_name}")
    
    # Đọc file JSON
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Danh sách để lưu kết quả so sánh
    comparisons = []
    missing_data = []
    
    # Xử lý dữ liệu
    
    for map_name, query_name in valid_maps:
        query_key = f"{map_name}_{query_name}"
        
        if query_key not in results:
            missing_data.append(query_key)
            continue
        
        query_result = results[query_key]
        methods = query_result.get('methods', {})
        
        # Lấy độ dài đường đi từ A* (vẫn lấy từ JSON)
        astar_data = methods.get('A*', {})
        astar_length = astar_data.get('length')

        # Lấy checkpoints từ VLM-A* và tính độ dài thực tế
        llm_astar_data = methods.get('VLM-A*', {})
        llm_checkpoints = llm_astar_data.get('checkpoints')
        
        if astar_length is None or not llm_checkpoints:
            missing_data.append(query_key)
            continue
        
        # Tính độ dài thực tế từ checkpoints của LLM-A*
        llm_astar_length = calculate_path_length_from_checkpoints(llm_checkpoints)
        
        if llm_astar_length == 0:
            missing_data.append(query_key)
            continue
        
        # Tính tỷ lệ phần trăm (LLM-A* / A* * 100)
        percentage = (llm_astar_length / astar_length) * 100
        
        comparisons.append({
            'query_key': query_key,
            'astar_length': astar_length,
            'llm_astar_length': llm_astar_length,
            'percentage': percentage
        })
        
        # Không in chi tiết từng map
    
    # Tính toán thống kê tổng hợp
    if comparisons:
        percentages = [comp['percentage'] for comp in comparisons]
        avg_percentage = sum(percentages) / len(percentages)
        min_percentage = min(percentages)
        max_percentage = max(percentages)
        
        # Đếm số map tốt hơn, bằng và kém hơn A*
        better_count = sum(1 for p in percentages if p < 100)
        equal_count = sum(1 for p in percentages if p == 100)
        worse_count = sum(1 for p in percentages if p > 100)
        
        print(f"Hiệu suất LLM-A* so với A*: {avg_percentage:.2f}%")
    
    return {
        'total_valid_maps': len(valid_maps),
        'analyzed_maps': len(comparisons),
        'missing_data': missing_data,
        'comparisons': comparisons,
        'average_percentage': avg_percentage if comparisons else None,
        'min_percentage': min_percentage if comparisons else None,
        'max_percentage': max_percentage if comparisons else None,
        'better_count': better_count if comparisons else 0,
        'equal_count': equal_count if comparisons else 0,
        'worse_count': worse_count if comparisons else 0
    }


def main():
    """Hàm main"""
    # Đường dẫn đến các file
    current_dir = Path(__file__).parent.parent
    json_file = current_dir / "Ly/dataset_results_qwen_qwen_20250925_154243.json"
    valid_path_file = current_dir / "valid_path.txt"
    
    # Kiểm tra các file có tồn tại không
    if not json_file.exists():
        print(f"❌ Không tìm thấy file JSON: {json_file}")
        return
    
    if not valid_path_file.exists():
        print(f"❌ Không tìm thấy file valid_path.txt: {valid_path_file}")
        return
    
    # Đọc dữ liệu
    
    # Thực hiện phân tích
    results = calculate_path_length_comparison(str(json_file), str(valid_path_file))
    
    # Kết thúc


if __name__ == "__main__":
    main()