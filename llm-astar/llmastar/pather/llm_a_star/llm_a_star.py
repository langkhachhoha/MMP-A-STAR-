import json
import math
import heapq

from llmastar.env.search import env, plotting
from llmastar.model import ChatGPT, Llama3, LLAMA3_FPT, QWEN, DeepSeek
from llmastar.utils import is_lines_collision, list_parse
from .prompt import *

class LLMAStar:
    """LLM-A* algorithm with cost + heuristics as the priority."""
    
    GPT_METHOD = "PARSE"
    GPT_LLMASTAR_METHOD = "LLM-A*"

    def __init__(self, llm='gpt', prompt='standard', improved=False, adaptive=False, alpha_decay=0.9):
        self.llm = llm
        self.improved = improved  
        self.adaptive = adaptive  
        self.alpha_decay = alpha_decay  
        self.alpha_factor = 1.0  
        self.target_update_count = 0  
        
        if self.llm == 'gpt':
            self.parser = ChatGPT(method=self.GPT_METHOD, sysprompt=sysprompt_parse, example=example_parse)
            self.model = ChatGPT(method=self.GPT_LLMASTAR_METHOD, sysprompt="", example=None)
        elif self.llm == 'llama3_fpt':
            self.parser = LLAMA3_FPT(method=self.GPT_METHOD, sysprompt=sysprompt_parse, example=example_parse)
            self.model = LLAMA3_FPT(method=self.GPT_LLMASTAR_METHOD, sysprompt="", example=None)
        elif self.llm == 'qwen':
            self.parser = QWEN(method=self.GPT_METHOD, sysprompt=sysprompt_parse, example=example_parse)
            self.model = QWEN(method=self.GPT_LLMASTAR_METHOD, sysprompt="", example=None)
        elif self.llm == 'deepseek':
            self.parser = DeepSeek(method=self.GPT_METHOD, sysprompt=sysprompt_parse, example=example_parse)
            self.model = DeepSeek(method=self.GPT_LLMASTAR_METHOD, sysprompt="", example=None)
        else:
            print(self.llm)
            raise ValueError("Invalid LLM model. Choose 'gpt', 'llama', 'llama3_fpt', 'qwen', or 'deepseek'.")

        assert prompt in ['standard', 'cot', 'repe'], "Invalid prompt type. Choose 'standard', 'cot', or 'repe'."
        self.prompt = prompt

    def _parse_query(self, query):
        """Parse input query using the specified LLM model."""
        if isinstance(query, str):
            if self.llm != '':
                response = self.parser.chat(query)
                return json.loads(response)
            else:
                print(self.llm)
                raise ValueError("Invalid LLM model.")
        return query

    def _initialize_parameters(self, input_data):
        """Initialize environment parameters from input data."""
        self.s_start = tuple(input_data['start'])
        self.s_goal = tuple(input_data['goal'])
        self.horizontal_barriers = input_data['horizontal_barriers']
        self.vertical_barriers = input_data['vertical_barriers']
        # Tạo copy để không modify input_data gốc
        self.range_x = input_data['range_x'].copy()
        self.range_y = input_data['range_y'].copy()
        self.Env = env.Env(self.range_x[1], self.range_y[1], self.horizontal_barriers, self.vertical_barriers)
        self.plot = plotting.Plotting(self.s_start, self.s_goal, self.Env)
        # Adjust range limits
        self.range_x[1] -= 1
        self.range_y[1] -= 1
        self.u_set = self.Env.motions
        self.obs = self.Env.obs
        self.OPEN = []
        self.CLOSED = []
        self.PARENT = dict()
        self.g = dict()

    def _initialize_llm_paths(self):
        """Initialize paths using LLM suggestions."""
        start, goal = list(self.s_start), list(self.s_goal)
        query = self._generate_llm_query(start, goal)

        if self.llm != '':
            response = self.model.ask(prompt=query, max_tokens=2048)
        else:
            raise ValueError("Invalid LLM model.")

        nodes = list_parse(response + "]]")
        self.target_list = self._filter_valid_nodes(nodes)

        if not self.target_list or self.target_list[0] != self.s_start:
            self.target_list.insert(0, self.s_start)
        if not self.target_list or self.target_list[-1] != self.s_goal:
            self.target_list.append(self.s_goal)
        
        self.original_target_list = self.target_list.copy()
        self.i = 1
        self.s_target = self.target_list[1]

    def _generate_llm_query(self, start, goal):
        """Generate the query for the LLM."""
        if self.llm != '':
            return gpt_prompt[self.prompt].format(start=start, goal=goal,
                                horizontal_barriers=self.horizontal_barriers,
                                vertical_barriers=self.vertical_barriers)
        elif self.llm == 'llama':
            return llama_prompt[self.prompt].format(start=start, goal=goal,
                                    horizontal_barriers=self.horizontal_barriers,
                                    vertical_barriers=self.vertical_barriers)

    def _filter_valid_nodes(self, nodes):
        """Filter out invalid nodes based on environment constraints.
        If a node is invalid, replace it with a valid neighbor if possible."""
        valid_nodes = []
        for node in nodes:
            n = (node[0], node[1])
            if n not in self.obs and self.range_x[0] + 1 < n[0] < self.range_x[1] - 1 and self.range_y[0] + 1 < n[1] < self.range_y[1] - 1:
                valid_nodes.append(n)
            else:
                found = False
                for u in self.u_set:
                    neighbor = (n[0] + u[0], n[1] + u[1])
                    if neighbor not in self.obs and self.range_x[0] + 1 < neighbor[0] < self.range_x[1] - 1 and self.range_y[0] + 1 < neighbor[1] < self.range_y[1] - 1:
                        valid_nodes.append(neighbor)
                        found = True
                        break
                if not found:
                    pass
        return valid_nodes

    def searching(self, query, filepath='temp.png', no_plot=False):
        """
        A* searching algorithm.
        :return: Path and search metrics.
        """
        self.filepath = filepath
        self.no_plot = no_plot
        self._initialize_parameters(query)
        self._initialize_llm_paths()
        
        return self._run_astar_search()
    
    def searching_with_predefined_targets(self, query, target_list, filepath='temp.png', no_plot=False):
        """
        Similar to searching but use predefined target_list from another LLM run
        """
        self.filepath = filepath
        self.no_plot = no_plot
        self._initialize_parameters(query)
        
        self.original_target_list = target_list
        self.target_list = target_list.copy()
        
        self.i = 1
        if len(self.target_list) > 1:
            self.s_target = self.target_list[1]
        else:
            self.s_target = self.s_goal
            
        return self._run_astar_search()
    
    def _run_astar_search(self):
        """
        Chạy thuật toán A* search
        """
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))

        path_found = False
        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                path_found = True
                break

            if self.improved and self._should_update_target_improved(s):
                self._update_target_improved()
                self._update_queue()

            for s_n in self.get_neighbor(s):
                if not self.improved and s_n == self.s_target and self.s_goal != self.s_target:
                    self._update_target()
                    self._update_queue()
                    
                if s_n in self.CLOSED:
                    continue

                new_cost = self.g[s] + self.cost(s, s_n)
                if s_n not in self.g:
                    self.g[s_n] = math.inf
                    
                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        if path_found:
            path = self.extract_path(self.PARENT)
        else:
            path = []  
            print("⚠️ Không tìm được đường đi từ start đến goal!")
            
        visited = self.CLOSED
        
        path_length = 0
        if path and len(path) > 1:
            path_length = sum(self._euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1))
            
        result = {
            "operation": len(self.CLOSED),
            "storage": len(self.g),
            "length": path_length,
            "llm_output": self.target_list,
            "path_found": path_found
        }
        print(result)
        
        if not self.no_plot:
            method_name = "LLM-A*"
            if self.adaptive:
                method_name = f"LLM-A* (Adaptive α={self.alpha_decay})"
            self.plot.animation(path, visited, False, method_name, self.filepath, waypoints=self.target_list)
            
            clean_filepath = self.filepath.replace('.png', '_waypoints.png')
            self.plot.plot_clean_map_with_waypoints(f"{method_name} Waypoints", clean_filepath, waypoints=self.target_list)
            
            clean_map_filepath = self.filepath.replace('.png', '_clean_map.png')
            self.plot.plot_clean_map_with_waypoints(f"{method_name} Clean Map", clean_map_filepath, waypoints=None)
        
        return result

    @staticmethod
    def _euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _update_queue(self):
        queue = []
        for _, s in self.OPEN:
            heapq.heappush(queue, (self.f_value(s), s))
        self.OPEN = queue

    def _update_target(self):
        """Update the current target in the path."""
        self.i += 1
        if self.i < len(self.target_list):
            self.s_target = self.target_list[self.i]
            
            if self.adaptive:
                self.target_update_count += 1
                self.alpha_factor = self.alpha_decay ** self.target_update_count

    def _should_update_target_improved(self, s):
        """
        """
        if not self.improved or self.s_target == self.s_goal:
            return False
            
        dist_to_target = self._euclidean_distance(s, self.s_target)
        dist_to_goal = self._euclidean_distance(s, self.s_goal)
        
        return dist_to_target > dist_to_goal

    def _update_target_improved(self):
        while self.i + 1 < len(self.target_list):
            self.i += 1
            self.s_target = self.target_list[self.i]
            if self.s_target == self.s_goal:
                break

    def get_neighbor(self, s):
        """Find neighbors of state s that are not in obstacles."""
        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """Calculate cost for the motion from s_start to s_goal."""
        return math.inf if self.is_collision(s_start, s_goal) else math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """Check if the line segment (s_start, s_end) collides with any barriers."""
        line1 = [s_start, s_end]
        return any(is_lines_collision(line1, [[h[1], h[0]], [h[2], h[0]]]) for h in self.horizontal_barriers) or \
                any(is_lines_collision(line1, [[v[0], v[1]], [v[0], v[2]]]) for v in self.vertical_barriers) or \
                any(is_lines_collision(line1, [[x, self.range_y[0]], [x, self.range_y[1]]]) for x in self.range_x) or \
                any(is_lines_collision(line1, [[self.range_x[0], y], [self.range_x[1], y]]) for y in self.range_y)

    def f_value(self, s):
        """Compute the f-value for state s."""
        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """Extract the path based on the PARENT set."""
        if self.s_goal not in PARENT:
            return []  
            
        path = [self.s_goal]
        current = self.s_goal
        
        while current != self.s_start:
            if current not in PARENT:
                return []  
            current = PARENT[current]
            path.append(current)
            
        return path[::-1]

    def heuristic(self, s):
        """Calculate heuristic value with adaptive decay."""
        goal_distance = math.hypot(self.s_goal[0] - s[0], self.s_goal[1] - s[1])
        target_distance = math.hypot(self.s_target[0] - s[0], self.s_target[1] - s[1])
        
        if self.adaptive:
            target_distance *= self.alpha_factor
            
        return goal_distance + target_distance

