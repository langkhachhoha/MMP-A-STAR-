import os
import json
import matplotlib.pyplot as plt
from llmastar.env.search import env, plotting
import random

def create_base_maps():
    """Tạo 10 base maps với các cấu hình barriers khác nhau"""
    base_maps = []
    
    # Map 1: Simple maze
    base_maps.append({
    "size": [101, 61],
    "horizontal_barriers": [
        [15, 10, 40],   # tường ngang 1
        [15, 50, 90],   # tường ngang 2 - tạo khoảng trống ở giữa
        [25, 0, 25],    # tường ngang 3
        [25, 35, 60],   # tường ngang 4
        [35, 15, 45],   # tường ngang 5
        [35, 70, 100],  # tường ngang 6
        [45, 5, 35],    # tường ngang 7
        [45, 55, 85],   # tường ngang 8
        [55, 20, 50],   # tường ngang 9
        [55, 65, 95]    # tường ngang 10 (từ x=65 đến x=95 bao gồm cả 95)
    ],
    "vertical_barriers": [
        [20, 5, 20],    # tường dọc 1
        [20, 30, 50],   # tường dọc 2
        [40, 15, 30],   # tường dọc 3
        [40, 40, 60],   # tường dọc 4
        [60, 5, 15],    # tường dọc 5
        [60, 25, 40],   # tường dọc 6
        [60, 50, 60],   # tường dọc 7
        [80, 10, 35],   # tường dọc 8
        [80, 45, 60],   # tường dọc 9
        [30, 35, 45],   # tường dọc 10
        [70, 20, 35],   # tường dọc 11
        [90, 15, 45]    # tường dọc 12
    ],
    "range_x": [0, 101],
    "range_y": [0, 61]
    })
    
    # Map 2: Complex maze with gaps
    base_maps.append({
  "horizontal_barriers": [
    [10, 5, 25],
    [10, 35, 65],
    [10, 75, 100],
    [20, 15, 45],
    [20, 60, 80],
    [30, 0, 20],
    [30, 40, 70],
    [40, 10, 35],
    [40, 55, 85],
    [50, 25, 55]
  ],
  "vertical_barriers": [
    [15, 10, 30],
    [25, 30, 45],
    [35, 5, 25],
    [35, 35, 55],
    [50, 15, 40],
    [65, 10, 30],
    [65, 45, 60],
    [80, 15, 40],
    [90, 5, 25],
    [90, 35, 55],
    [45, 45, 51],
    [70, 20, 30]
  ],
  "range_x": [0, 101],
  "range_y": [0, 61]
    })
    
    # Map 3: Zigzag pattern
    base_maps.append({
  "horizontal_barriers": [
    [15, 10, 40],
    [15, 60, 95],
    [25, 0, 30],
    [25, 45, 70],
    [35, 75, 100],
    [45, 10, 35],
    [45, 55, 70],
    [55, 5, 25],
    [55, 40, 70]
  ],
  "vertical_barriers": [
    [20, 5, 25],
    [20, 35, 55],
    [30, 15, 40],
    [40, 10, 20],
    [40, 35, 60],
    [55, 10, 30],
    [55, 45, 60],
    [80, 25, 50],
    [95, 5, 25],
    [95, 35, 55]
  ],
  "range_x": [0, 101],
  "range_y": [0, 61]
    })
    
    # Map 4: Corridor style - tương tự main.py
    base_maps.append({
  "horizontal_barriers": [
    [10, 20, 55],
    [10, 70, 95],
    [20, 0, 25],
    [20, 40, 65],
    [30, 15, 35],
    [30, 75, 100],
    [40, 5, 30],
    [40, 60, 75],
    [50, 20, 55],
    [55, 70, 90]
  ],
  "vertical_barriers": [
    [15, 15, 30],
    [30, 20, 50],
    [45, 5, 25],
    [45, 40, 55],
    [60, 10, 35],
    [60, 45, 60],
    [75, 15, 40],
    [85, 5, 25],
    [85, 35, 55],
    [95, 15, 45],
    [35, 25, 45]
  ],
  "range_x": [0, 101],
  "range_y": [0, 61]
    })
    
    # Map 5: Sparse obstacles
    base_maps.append({
  "horizontal_barriers": [
    [15, 5, 30],
    [15, 40, 70],
    [15, 90, 100],
    [25, 20, 30],
    [25, 60, 85],
    [35, 10, 25],
    [35, 55, 75],
    [45, 10, 40],
    [45, 75, 95],
    [55, 25, 40]
  ],
  "vertical_barriers": [
    [20, 10, 35],
    [30, 5, 25],
    [30, 35, 55],
    [50, 10, 30],
    [50, 45, 60],
    [65, 5, 25],
    [65, 40, 60],
    [80, 15, 45],
    [90, 0, 20],
    [90, 30, 50],
    [40, 25, 45]
  ],
  "range_x": [0, 101],
  "range_y": [0, 61]
    })

  # map 6
    base_maps.append({
    "size": [101, 61],
    "horizontal_barriers": [
        [15, 10, 40],   # tường ngang 1
        [15, 50, 90],   # tường ngang 2 - tạo khoảng trống ở giữa
        [25, 0, 25],    # tường ngang 3
        [25, 35, 50],   # tường ngang 4
        [35, 15, 45],   # tường ngang 5
        [35, 80, 100],  # tường ngang 6
        [45, 5, 35],    # tường ngang 7
        [45, 55, 85],   # tường ngang 8
        [55, 20, 50],   # tường ngang 9
        [55, 65, 95]    # tường ngang 10 (từ x=65 đến x=95 bao gồm cả 95)
    ],
    "vertical_barriers": [
        [20, 5, 20],    # tường dọc 1
        [20, 30, 50],   # tường dọc 2
        [40, 15, 30],   # tường dọc 3
        [40, 40, 60],   # tường dọc 4
        [60, 5, 15],    # tường dọc 5
        [60, 15, 35],   # tường dọc 6
        [60, 50, 60],   # tường dọc 7
        [80, 10, 35],   # tường dọc 8
        [80, 45, 60],   # tường dọc 9
        [30, 35, 45],   # tường dọc 10
        [70, 20, 35],   # tường dọc 11
        [90, 15, 45]    # tường dọc 12
    ],
    "range_x": [0, 101],
    "range_y": [0, 61]
    })


    # Map 2: Complex maze with gaps
    base_maps.append({
  "horizontal_barriers": [
    [10, 5, 25],
    [10, 35, 65],
    [10, 75, 100],
    [20, 15, 25],
    [20, 60, 80],
    [30, 0, 20],
    [30, 40, 70],
    [40, 10, 35],
    [40, 55, 85],
    [50, 25, 55]
  ],
  "vertical_barriers": [
    [15, 10, 30],
    [25, 30, 45],
    [35, 5, 25],
    [35, 35, 55],
    [50, 15, 40],
    [65, 10, 30],
    [65, 45, 60],
    [80, 15, 40],
    [90, 5, 15],
    [90, 35, 55],
    [45, 45, 51],
    [70, 20, 30]
  ],
  "range_x": [0, 101],
  "range_y": [0, 61]
    })
    
    # Map 3: Zigzag pattern
    base_maps.append({
  "horizontal_barriers": [
    [15, 10, 40],
    [15, 60, 95],
    [25, 0, 30],
    [25, 45, 70],
    [35, 85, 100],
    [45, 10, 35],
    [45, 55, 70],
    [55, 5, 25],
    [55, 40, 70]
  ],
  "vertical_barriers": [
    [20, 5, 25],
    [20, 35, 55],
    [30, 15, 40],
    [40, 10, 20],
    [40, 35, 60],
    [55, 10, 30],
    [55, 45, 60],
    [80, 25, 50],
    [95, 5, 25],
    [95, 35, 55],
    [65, 25, 35],
  ],
  "range_x": [0, 101],
  "range_y": [0, 61]
    })
    
    # Map 4: Corridor style - tương tự main.py
    base_maps.append({
  "horizontal_barriers": [
    [10, 20, 55],
    [10, 70, 95],
    [20, 0, 25],
    [20, 40, 65],
    [30, 15, 35],
    [30, 75, 100],
    [40, 5, 30],
    [50, 20, 55],
    [55, 70, 90]
  ],
  "vertical_barriers": [
    [50, 30, 40],
    [15, 15, 30],
    [45, 5, 25],
    [45, 40, 55],
    [60, 10, 35],
    [60, 45, 60],
    [75, 15, 40],
    [85, 5, 25],
    [85, 35, 55],
    [95, 15, 45],
    [35, 25, 45]
  ],
  "range_x": [0, 101],
  "range_y": [0, 61]
    })
    
    # Map 5: Sparse obstacles
    base_maps.append({
  "horizontal_barriers": [
    [15, 5, 30],
    [15, 40, 70],
    [15, 90, 100],
    [25, 20, 30],
    [25, 60, 85],
    [35, 10, 25],
    [35, 55, 75],
    [45, 0, 40],
    [45, 75, 95],
    [55, 25, 40],
    [5, 40, 60]
  ],
  "vertical_barriers": [
    [20, 10, 35],
    [30, 5, 25],
    [30, 35, 55],
    [50, 45, 60],
    [65, 5, 25],
    [65, 40, 60],
    [80, 15, 35],
    [90, 0, 20],
    [90, 30, 50],
    [40, 25, 45]
  ],
  "range_x": [0, 101],
  "range_y": [0, 61]
    })
    
    
    return base_maps

def get_predefined_start_goal_pairs():
    """Trả về các cặp start-goal được định nghĩa sẵn cho từng map"""
    predefined_pairs = {
        1: [  # Map 1
            ([5, 5], [96, 55]),   
            ([96, 55], [5, 5]),

            ([5, 5], [75, 50]),
            ([75, 50], [5, 5]),

            ([5, 30], [96, 55]),
            ([96, 55], [5, 30]),

            ([25, 50], [80, 5]),
            ([80, 5], [25, 50]),

            ([10, 30], [70,50]),
            ([70,50], [10,30]),

            ([10, 58], [75, 25]),
            ([75, 25], [10, 58]),

            ([30, 30], [10, 55]),
            ([10, 55], [30, 30]),

            ([5, 5], [90, 10]),
            ([90, 10], [5, 5]),

            ([20, 27], [85, 40]),
            ([85, 40], [20, 27]),

            ([30, 50], [30, 5]),
            ([30, 5], [30, 50])
        ],
        2: [  # Map 2
            ([5, 5], [96, 55]),
            ([96, 55], [5, 5]),

            ([30, 25], [95, 40]),
            ([95, 40], [30, 25]),

            ([5, 40], [60, 15]),
            ([60, 15], [5, 40]),

            ([10, 35], [80,50]),
            ([80,50], [10,35]),

            ([20, 25], [80, 50]),
            ([80, 50], [20, 25]),

            ([25, 45], [70, 10]),
            ([70, 10], [25, 45]),

            ([30, 20], [60, 50]),
            ([60, 50], [30, 20]),

            ([40, 15], [55, 40]),
            ([55, 40], [40, 15]),

            ([20, 55], [85, 35]),
            ([85, 35], [20, 55]),

            ([35, 30], [65, 45]),
            ([65, 45], [35, 30]),
        ],
        3: [  # Map 3
            ([5, 10], [85, 50]),
            ([85, 50], [5, 10]),

            ([20, 30], [85, 50]),
            ([85, 50], [20, 30]),

            ([20, 30], [80, 20]),
            ([80, 20], [20, 30]),

            ([80, 20], [35, 40]),
            ([35, 40], [80, 20]),

            ([50, 30], [90, 10]),
            ([90, 10], [50, 30]),

            ([15, 35], [55, 15]),
            ([55, 15], [15, 35]),

            ([30, 25], [60, 50]),
            ([60, 50], [30, 25]),

            ([40, 15], [50, 40]),
            ([50, 40], [40, 15]),

            ([10, 45], [85, 20]),
            ([85, 20], [10, 45]),

            ([35, 30], [65, 55]),
            ([65, 55], [35, 30])
        ],
        4: [  # Map 4
            ([5, 5], [96, 55]),
            ([96, 55], [5, 5]),

            ([65, 25], [96, 55]),
            ([96, 55], [65, 25]),

            ([90, 40], [65, 25]),
            ([65, 25], [90, 40]),

            ([5,5], [70, 35]),
            ([70, 35], [5,5]),

            ([5, 5], [50, 25]),
            ([50, 25], [5, 5]),

                ([15, 35], [60, 15]),
    ([60, 15], [15, 35]),

    ([30, 25], [55, 50]),
    ([55, 50], [30, 25]),

    ([40, 15], [50, 40]),
    ([50, 40], [40, 15]),

    ([10, 45], [80, 20]),
    ([80, 20], [10, 45]),

    ([20, 28], [90, 5]),
    ([90, 5], [20, 28])
        ],
        5: [  # Map 5
            ([5, 10], [96, 50]),
            ([96, 50], [5, 10]),

            ([10, 40], [91, 20]),
            ([91, 20], [10, 40]),

            ([20, 5], [80, 55]),
            ([80, 55], [20, 5]),

            ([20, 50], [96, 50]),
            ([96, 50], [20, 50]),

            ([10, 40], [60, 5]),
            ([60, 5], [10, 40]),

            ([15, 35], [60, 15]),
            ([60, 15], [15, 35]),

            ([30, 25], [55, 50]),
            ([55, 50], [30, 25]),

            ([20, 5], [60, 55]),
            ([60, 55], [20, 5]),

            ([10, 45], [80, 20]),
            ([80, 20], [10, 45]),

            ([35, 30], [65, 45]),
            ([65, 45], [35, 30])
        ]
    }
    return predefined_pairs

def generate_start_goal_pairs(base_map, map_idx):
    """Lấy các cặp start-goal được định nghĩa sẵn cho map"""
    predefined_pairs = get_predefined_start_goal_pairs()
    
    return predefined_pairs[(map_idx-1) % 5 + 1]
    # else:
    #     # Fallback nếu không có định nghĩa sẵn
    #     return [([5, 5], [95, 55]), ([10, 10], [90, 50])]

def is_point_valid(point, obstacles, range_x=None, range_y=None):
    """Kiểm tra xem một điểm có hợp lệ (trong bounds và không nằm trên obstacle) hay không"""
    x, y = point
    
    # Kiểm tra bounds nếu được cung cấp
    if range_x is not None and range_y is not None:
        if not (range_x[0] <= x < range_x[1] and range_y[0] <= y < range_y[1]):
            return False
    
    # Kiểm tra obstacles
    return tuple(point) not in obstacles

def find_valid_point_nearby(point, obstacles, range_x, range_y, max_radius=5):
    """Tìm điểm hợp lệ gần nhất bằng cách thử 8 hướng xung quanh"""
    x, y = point
    
    # Nếu điểm đã hợp lệ thì trả về luôn
    if is_point_valid(point, obstacles, range_x, range_y):
        return point
    
    # 8 hướng: lên, xuống, trái, phải, và 4 hướng chéo
    directions = [
        (0, 1),   # lên
        (0, -1),  # xuống  
        (-1, 0),  # trái
        (1, 0),   # phải
        (-1, 1),  # trái-lên
        (1, 1),   # phải-lên
        (-1, -1), # trái-xuống
        (1, -1)   # phải-xuống
    ]
    
    # Thử từng bán kính từ 1 đến max_radius
    for radius in range(1, max_radius + 1):
        for dx, dy in directions:
            new_x = x + dx * radius
            new_y = y + dy * radius
            
            # Kiểm tra trong bounds
            if (range_x[0] <= new_x < range_x[1] and 
                range_y[0] <= new_y < range_y[1]):
                
                new_point = [new_x, new_y]
                if is_point_valid(new_point, obstacles, range_x, range_y):
                    print(f"🔄 Moved point {point} -> {new_point} (distance: {radius})")
                    return new_point
    
    # Nếu không tìm được điểm hợp lệ
    print(f"❌ Could not find valid point near {point} within radius {max_radius}")
    return point

def validate_and_fix_start_goal_points(start, goal, horizontal_barriers, vertical_barriers, range_x, range_y):
    """Kiểm tra và tự động sửa start và goal points nếu không hợp lệ"""
    # Tạo environment để lấy obstacles
    Env = env.Env(range_x[1], range_y[1], horizontal_barriers, vertical_barriers)
    obstacles = Env.obs_map()
    
    original_start = start.copy()
    original_goal = goal.copy()
    
    start_valid = is_point_valid(start, obstacles, range_x, range_y)
    goal_valid = is_point_valid(goal, obstacles, range_x, range_y)
    
    # Tự động fix start point nếu không hợp lệ
    if not start_valid:
        print(f"⚠️  WARNING: Start point {start} is on an obstacle!")
        fixed_start = find_valid_point_nearby(start, obstacles, range_x, range_y)
        start[:] = fixed_start  # Update in-place
        start_valid = True
    
    # Tự động fix goal point nếu không hợp lệ
    if not goal_valid:
        print(f"⚠️  WARNING: Goal point {goal} is on an obstacle!")
        fixed_goal = find_valid_point_nearby(goal, obstacles, range_x, range_y)
        goal[:] = fixed_goal  # Update in-place
        goal_valid = True
        
    return start_valid, goal_valid, (start != original_start or goal != original_goal)

def create_map_image(query, output_path):
    """Tạo hình ảnh cho một map"""
    # Kiểm tra và tự động sửa start và goal trước khi vẽ
    start_valid, goal_valid, was_fixed = validate_and_fix_start_goal_points(
        query["start"], query["goal"], 
        query["horizontal_barriers"], query["vertical_barriers"],
        query["range_x"], query["range_y"]
    )
    
    Env = env.Env(query["range_x"][1], query["range_y"][1], 
                  query["horizontal_barriers"], query["vertical_barriers"])
    plot = plotting.Plotting(tuple(query["start"]), tuple(query["goal"]), Env)
    
    plt.clf()
    plot.plot_grid(f"Map: Start{query['start']} -> Goal{query['goal']}")
    
    # Vẽ start và goal với màu sắc khác nhau dựa trên tính hợp lệ
    if start_valid:
        plt.plot(query["start"][0], query["start"][1], "bs", markersize=12, label="Start")
    else:
        plt.plot(query["start"][0], query["start"][1], "rs", markersize=12, label="Start (INVALID)")
        
    if goal_valid:
        plt.plot(query["goal"][0], query["goal"][1], "gs", markersize=12, label="Goal")
    else:
        plt.plot(query["goal"][0], query["goal"][1], "rs", markersize=12, label="Goal (INVALID)")
    
    plt.legend()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def validate_all_predefined_pairs():
    """Kiểm tra tính hợp lệ của tất cả predefined start-goal pairs"""
    print("🔍 Validating all predefined start-goal pairs...")
    base_maps = create_base_maps()
    predefined_pairs = get_predefined_start_goal_pairs()
    
    total_pairs = 0
    invalid_pairs = 0
    
    for map_idx, base_map in enumerate(base_maps, 1):
        if map_idx in predefined_pairs:
            pairs = predefined_pairs[map_idx]
            for pair_idx, (start, goal) in enumerate(pairs, 1):
                total_pairs += 1
                start_valid, goal_valid, was_fixed = validate_and_fix_start_goal_points(
                    start, goal,
                    base_map["horizontal_barriers"], base_map["vertical_barriers"],
                    base_map["range_x"], base_map["range_y"]
                )
                
                if not (start_valid and goal_valid):
                    invalid_pairs += 1
                    print(f"❌ Map {map_idx}, Pair {pair_idx}: {start} -> {goal}")
    
    print(f"📊 Validation Summary: {total_pairs - invalid_pairs}/{total_pairs} pairs are valid")
    if invalid_pairs > 0:
        print(f"⚠️  Found {invalid_pairs} invalid pairs that need to be fixed!")
    else:
        print("✅ All predefined pairs are valid!")
    
    return invalid_pairs == 0

def create_complex_level_map(base_map, level):
    """
    Tạo map với độ phức tạp theo level
    Level 1: 2 horizontal + 2 vertical barriers
    Level 2: 4 horizontal + 4 vertical barriers  
    Level 3: 6 horizontal + 6 vertical barriers
    Level 4: 8 horizontal + 8 vertical barriers
    Level 5: Toàn bộ barriers (base map)
    """
    if level == 5:
        # Level 5 là base map đầy đủ
        return base_map.copy()
    
    # Tính số barrier cần thiết cho level
    num_barriers = level * 2
    
    # Lấy barriers từ base map
    h_barriers = base_map["horizontal_barriers"].copy()
    v_barriers = base_map["vertical_barriers"].copy()
    
    # Xử lý trường hợp không đủ barrier
    available_h = len(h_barriers)
    available_v = len(v_barriers)
    
    # Điều chỉnh số barrier nếu không đủ
    actual_h_barriers = min(num_barriers, available_h)
    actual_v_barriers = min(num_barriers, available_v)
    
    # Nếu vẫn không đủ, cảnh báo
    if actual_h_barriers < num_barriers or actual_v_barriers < num_barriers:
        print(f"⚠️  Level {level}: Adjusted barriers - H:{actual_h_barriers}/{num_barriers}, V:{actual_v_barriers}/{num_barriers}")
    
    # Chọn ngẫu nhiên barriers
    selected_h_barriers = random.sample(h_barriers, actual_h_barriers)
    selected_v_barriers = random.sample(v_barriers, actual_v_barriers)
    
    # Tạo map mới
    complex_map = base_map.copy()
    complex_map["horizontal_barriers"] = selected_h_barriers
    complex_map["vertical_barriers"] = selected_v_barriers
    
    return complex_map

def create_complex_dataset():
    """Tạo Complex_Dataset với 5 level độ phức tạp"""
    print("🏗️  Creating Complex_Dataset...")
    
    # Tạo thư mục Complex_Dataset
    complex_dataset_dir = "Complex_Dataset"
    if not os.path.exists(complex_dataset_dir):
        os.makedirs(complex_dataset_dir)
    
    base_maps = create_base_maps()
    predefined_pairs = get_predefined_start_goal_pairs()
    
    # Tạo 5 level
    for level in range(1, 6):
        level_dir = os.path.join(complex_dataset_dir, f"level_{level}")
        if not os.path.exists(level_dir):
            os.makedirs(level_dir)
        
        print(f"\n📊 Creating Level {level} (Target: {level*2} barriers each type)...")
        
        # Tạo 40 map cho mỗi level
        for map_count in range(1, 41):
            # Chọn ngẫu nhiên một base map
            base_map_idx = random.randint(0, len(base_maps) - 1)
            base_map = base_maps[base_map_idx]
            
            # Tạo complex map theo level
            complex_map = create_complex_level_map(base_map, level)
            
            # Chọn ngẫu nhiên một cặp start-goal từ base map tương ứng
            map_key = (base_map_idx % 5) + 1  # Map key từ 1-5
            if map_key in predefined_pairs:
                available_pairs = predefined_pairs[map_key]
                start, goal = random.choice(available_pairs)
            else:
                # Fallback nếu không có predefined pairs
                start, goal = [5, 5], [95, 55]
            
            # Tạo query
            query = complex_map.copy()
            query["start"] = list(start)
            query["goal"] = list(goal)
            
            # Kiểm tra và tự động sửa start và goal
            start_valid, goal_valid, was_fixed = validate_and_fix_start_goal_points(
                list(start), list(goal),
                complex_map["horizontal_barriers"], complex_map["vertical_barriers"],
                complex_map["range_x"], complex_map["range_y"]
            )
            
            # Cập nhật query với điểm đã được sửa (nếu có)
            if was_fixed:
                query["start"] = list(start)
                query["goal"] = list(goal)
            
            # Lưu query dưới dạng JSON
            query_path = os.path.join(level_dir, f"query_{map_count}.json")
            with open(query_path, 'w') as f:
                json.dump(query, f, indent=2)
            
            # Tạo hình ảnh
            image_path = os.path.join(level_dir, f"map_{map_count}.png")
            create_map_image(query, image_path)
            
            # In thông tin
            h_count = len(complex_map["horizontal_barriers"])
            v_count = len(complex_map["vertical_barriers"])
            status = "✅" if (start_valid and goal_valid) else "❌"
            
            if map_count % 10 == 0:  # In mỗi 10 map
                print(f"  {status} Created {map_count}/40 maps - H:{h_count}, V:{v_count} barriers")
        
        print(f"✅ Level {level} completed: 40 maps created")
    
    print(f"\n🎉 Complex_Dataset creation completed!")
    print(f"📁 Created 5 levels × 40 maps = 200 total maps")

def create_dataset():
    """Tạo toàn bộ dataset"""
    # Kiểm tra tính hợp lệ của tất cả predefined pairs trước
    if not validate_all_predefined_pairs():
        print("❌ Please fix invalid pairs before creating dataset!")
        return
        
    # Tạo thư mục Dataset
    dataset_dir = "Dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    base_maps = create_base_maps()
    
    for map_idx, base_map in enumerate(base_maps, 1):
        # Tạo thư mục cho map
        map_dir = os.path.join(dataset_dir, f"map_{map_idx}")
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        
        print(f"Creating map_{map_idx}...")
        
        # Tạo 5 cặp start-goal từ danh sách định nghĩa sẵn
        pairs = generate_start_goal_pairs(base_map, map_idx)
        
        for sub_idx, (start, goal) in enumerate(pairs, 1):
            # Tạo query
            query = base_map.copy()
            query["start"] = list(start)
            query["goal"] = list(goal)
            
            # Kiểm tra và tự động sửa start và goal
            start_valid, goal_valid, was_fixed = validate_and_fix_start_goal_points(
                start, goal,
                base_map["horizontal_barriers"], base_map["vertical_barriers"],
                base_map["range_x"], base_map["range_y"]
            )
            
            # Cập nhật query với điểm đã được sửa
            query["start"] = list(start)
            query["goal"] = list(goal)
            
            # Lưu query dưới dạng JSON
            query_path = os.path.join(map_dir, f"query_{sub_idx}.json")
            with open(query_path, 'w') as f:
                json.dump(query, f, indent=2)
            
            # Tạo hình ảnh
            image_path = os.path.join(map_dir, f"map_{sub_idx}.png")
            create_map_image(query, image_path)
            
            status = "✅" if (start_valid and goal_valid) else "❌"
            print(f"  {status} Created query_{sub_idx}: {start} -> {goal}")
    

def scale_barrier(barrier, scale_x, scale_y, is_horizontal):
    """
    Scale một barrier theo tỷ lệ
    barrier: [position, start, end]
    is_horizontal: True nếu là horizontal barrier, False nếu là vertical barrier
    """
    if is_horizontal:
        # Horizontal barrier: [y_position, x_start, x_end]
        return [
            int(barrier[0] * scale_y),  # scale y position
            int(barrier[1] * scale_x),  # scale x start
            int(barrier[2] * scale_x)   # scale x end
        ]
    else:
        # Vertical barrier: [x_position, y_start, y_end]
        return [
            int(barrier[0] * scale_x),  # scale x position
            int(barrier[1] * scale_y),  # scale y start
            int(barrier[2] * scale_y)   # scale y end
        ]

def scale_map(base_map, target_size):
    """
    Scale một base map lên kích thước mới
    base_map: map gốc với size (101, 61)
    target_size: [width, height] mới
    """
    # Tính tỷ lệ scale
    original_width = 101
    original_height = 61
    scale_x = target_size[0] / original_width
    scale_y = target_size[1] / original_height
    
    # Scale barriers
    scaled_h_barriers = [
        scale_barrier(b, scale_x, scale_y, is_horizontal=True)
        for b in base_map["horizontal_barriers"]
    ]
    
    scaled_v_barriers = [
        scale_barrier(b, scale_x, scale_y, is_horizontal=False)
        for b in base_map["vertical_barriers"]
    ]
    
    # Scale start và goal nếu có
    scaled_map = {
        "horizontal_barriers": scaled_h_barriers,
        "vertical_barriers": scaled_v_barriers,
        "range_x": [0, target_size[0]],
        "range_y": [0, target_size[1]]
    }
    
    return scaled_map

def scale_point(point, scale_x, scale_y):
    """Scale một điểm theo tỷ lệ"""
    return [int(point[0] * scale_x), int(point[1] * scale_y)]

def create_resolution_dataset():
    """Tạo Resolution_Dataset với 5 level kích thước khác nhau"""
    print("🏗️  Creating Resolution_Dataset...")
    
    # Tạo thư mục Resolution_Dataset
    resolution_dataset_dir = "Resolution_Dataset"
    if not os.path.exists(resolution_dataset_dir):
        os.makedirs(resolution_dataset_dir)
    
    base_maps = create_base_maps()
    predefined_pairs = get_predefined_start_goal_pairs()
    
    # Định nghĩa kích thước cho từng level
    level_sizes = {
        1: (51, 31),    # Level 1: 50x30 (width+1, height+1 để match với range)
        2: (101, 61),   # Level 2: 100x60 (kích thước gốc)
        3: (201, 121),  # Level 3: 200x120
        4: (301, 181),  # Level 4: 300x180
        5: (401, 241)   # Level 5: 400x240
    }
    
    # Tạo 5 level
    for level in range(1, 6):
        level_dir = os.path.join(resolution_dataset_dir, f"level_{level}")
        if not os.path.exists(level_dir):
            os.makedirs(level_dir)
        
        target_size = level_sizes[level]
        print(f"\n📊 Creating Level {level} (Size: {target_size[0]-1}x{target_size[1]-1})...")
        
        # Tính tỷ lệ scale
        scale_x = target_size[0] / 101
        scale_y = target_size[1] / 61
        
        # Tạo 40 map cho mỗi level
        for map_count in range(1, 41):
            # Chọn ngẫu nhiên một base map
            base_map_idx = random.randint(0, len(base_maps) - 1)
            base_map = base_maps[base_map_idx]
            
            # Scale map lên kích thước mới
            scaled_map = scale_map(base_map, target_size)
            
            # Chọn ngẫu nhiên một cặp start-goal từ base map tương ứng
            map_key = (base_map_idx % 5) + 1  # Map key từ 1-5
            if map_key in predefined_pairs:
                available_pairs = predefined_pairs[map_key]
                original_start, original_goal = random.choice(available_pairs)
                
                # Scale start và goal points
                scaled_start = scale_point(original_start, scale_x, scale_y)
                scaled_goal = scale_point(original_goal, scale_x, scale_y)
            else:
                # Fallback nếu không có predefined pairs
                scaled_start = scale_point([5, 5], scale_x, scale_y)
                scaled_goal = scale_point([95, 55], scale_x, scale_y)
            
            # Tạo query
            query = scaled_map.copy()
            query["start"] = scaled_start
            query["goal"] = scaled_goal
            
            # Kiểm tra và tự động sửa start và goal
            start_valid, goal_valid, was_fixed = validate_and_fix_start_goal_points(
                scaled_start, scaled_goal,
                scaled_map["horizontal_barriers"], scaled_map["vertical_barriers"],
                scaled_map["range_x"], scaled_map["range_y"]
            )
            
            # Cập nhật query với điểm đã được sửa (nếu có)
            if was_fixed:
                query["start"] = scaled_start
                query["goal"] = scaled_goal
            
            # Lưu query dưới dạng JSON
            query_path = os.path.join(level_dir, f"query_{map_count}.json")
            with open(query_path, 'w') as f:
                json.dump(query, f, indent=2)
            
            # Tạo hình ảnh
            image_path = os.path.join(level_dir, f"map_{map_count}.png")
            create_map_image(query, image_path)
            
            # In thông tin
            h_count = len(scaled_map["horizontal_barriers"])
            v_count = len(scaled_map["vertical_barriers"])
            status = "✅" if (start_valid and goal_valid) else "❌"
            
            if map_count % 10 == 0:  # In mỗi 10 map
                print(f"  {status} Created {map_count}/40 maps - Size:{target_size[0]-1}x{target_size[1]-1}, H:{h_count}, V:{v_count} barriers")
        
        print(f"✅ Level {level} completed: 40 maps created")
    
    print(f"\n🎉 Resolution_Dataset creation completed!")
    print(f"📁 Created 5 levels × 40 maps = 200 total maps")

def test_complex_level_creation():
    """Test tạo complex level để kiểm tra logic"""
    print("🧪 Testing complex level creation...")
    
    base_maps = create_base_maps()
    
    # Test với map đầu tiên
    test_map = base_maps[0]
    print(f"Original map - H:{len(test_map['horizontal_barriers'])}, V:{len(test_map['vertical_barriers'])}")
    
    # Test từng level
    for level in range(1, 6):
        complex_map = create_complex_level_map(test_map, level)
        h_count = len(complex_map["horizontal_barriers"])
        v_count = len(complex_map["vertical_barriers"])
        expected = level * 2 if level < 5 else len(test_map['horizontal_barriers'])
        
        print(f"Level {level}: H:{h_count}, V:{v_count} (Expected: {expected if level < 5 else 'full'})")
    
    print("✅ Complex level creation test completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Chỉ test logic
        test_complex_level_creation()
    elif len(sys.argv) > 1 and sys.argv[1] == "complex":
        # Chỉ tạo complex dataset
        create_complex_dataset()
    elif len(sys.argv) > 1 and sys.argv[1] == "resolution":
        # Chỉ tạo resolution dataset
        create_resolution_dataset()
    else:
        # Tạo cả ba dataset
        create_dataset()
        create_complex_dataset()
        create_resolution_dataset()
