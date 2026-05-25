from .llm_a_star import LLMAStar
from .vlm_a_star import VLMAStar

class LLMVLMRunner:

    def __init__(self, llm='gpt', vlm='llama', prompt='standard', improved=False, alpha_decay=0.9):
        self.llm = llm
        self.vlm = vlm
        self.prompt = prompt
        self.improved = improved
        self.alpha_decay = alpha_decay
        
    
    def run_all_four(self, query, base_filename='method', no_plot=False):
        results = {}
        
        print("=== Step 1: Running LLM-A* (adaptive=False) ===")
        llm_astar = LLMAStar(llm=self.llm, prompt=self.prompt, improved=self.improved, 
                            adaptive=False)
        llm_result = llm_astar.searching(query, f'{base_filename}_llm.png', no_plot)
        results['LLM-A*'] = llm_result
        
        shared_waypoints = llm_astar.original_target_list
        
        print("=== Step 2: Running LLM-A* (adaptive=True) ===")
        llm_astar_adaptive = LLMAStar(llm=self.llm, prompt=self.prompt, improved=self.improved, 
                                     adaptive=True, alpha_decay=self.alpha_decay)
        llm_result_adaptive = llm_astar_adaptive.searching_with_predefined_targets(
            query, shared_waypoints, f'{base_filename}_llm_adaptive.png', no_plot)
        results['LLM-A* (Adaptive)'] = llm_result_adaptive
        
        print("=== Step 3: Running VLM-A* (adaptive=False) ===")
        vlm_astar = VLMAStar(llm=self.llm, vlm=self.vlm, prompt=self.prompt, improved=self.improved, 
                            adaptive=False, shared_target_list=shared_waypoints)
        vlm_result = vlm_astar.searching(query, f'{base_filename}_vlm.png', no_plot)
        results['VLM-A*'] = vlm_result
        
        vlm_filtered_waypoints = vlm_astar.target_list  
        
        print("=== Step 4: Running VLM-A* (adaptive=True) ===")
        vlm_astar_adaptive = VLMAStar(llm=self.llm, vlm=self.vlm, prompt=self.prompt, improved=self.improved, 
                                     adaptive=True, alpha_decay=self.alpha_decay)
        vlm_result_adaptive = vlm_astar_adaptive.searching_with_predefined_targets(
            query, vlm_filtered_waypoints, f'{base_filename}_vlm_adaptive.png', no_plot)
        results['VLM-A* (Adaptive)'] = vlm_result_adaptive
        
        return results
