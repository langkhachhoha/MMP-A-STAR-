import json
import base64
import os
from llmastar.pather.llm_a_star import LLMAStar
from together import Together
import json
import base64
from typing import List
from pydantic import BaseModel, Field
from together import Together
from typing import List
from pydantic import BaseModel, Field

class WaypointSelection(BaseModel):
    selected_waypoints: List[int] = Field(
        description=(
            """Ordered list of original waypoint IDs chosen as essential navigation markers. 
            Selection must be based on first observing the map globally without waypoints, 
            identifying which corridors are open or blocked, then deciding which waypoints 
            are truly useful to guide the robot along a clear, feasible route. 
            Only retain waypoints that provide necessary guidance; discard those that lie in blocked 
            or invalid paths. No new waypoints should be created."""
        )
    )
    final_reasoning: str = Field(
        description=(
            """Detailed reasoning of the decision: first describe the overall feasible routes observed from the map 
            before considering any waypoints, then explain why each chosen waypoint is necessary as a marker, 
            and why others are discarded. This must be explicit, factual, and tied to barrier layout, 
            not just a short summary."""
        )
    )

class VLMAStar(LLMAStar):
    """VLM-A* algorithm extending LLM-A* with VLM waypoint refinement."""

    def __init__(self, llm='gpt', vlm='llama', prompt='standard', improved=False, adaptive=False, alpha_decay=0.9, vlm_api_key=None, shared_target_list=None):
        super().__init__(llm, prompt, improved, adaptive, alpha_decay)
        self.vlm = vlm
        self.client = Together(api_key="tgp_v1_CteYztP5IW8C8ZIhn8OkOXdgRhx_SfGdD9gPLXF6XJc")
        self.original_target_list = []  
        self.vlm_result = {} 
        self.shared_target_list = shared_target_list 
        
    def _initialize_llm_paths(self):
        """Initialize paths using shared LLM results, then refine with VLM."""
        if self.shared_target_list is not None:
            # Dùng target_list đã có
            self.target_list = self.shared_target_list.copy()
            
            if self.adaptive:
                self.original_target_list = self.target_list.copy()
                self.vlm_result = {"selected_waypoints": [], "final_reasoning": "Using pre-filtered VLM waypoints for adaptive mode"}
                return
            else:
                pass
        else:
            super()._initialize_llm_paths()
        
        self.original_target_list = self.target_list.copy()
        
        waypoints_image_path, clean_map_path = self._create_vlm_input_images()
        
        self.vlm_result = self._refine_waypoints_with_vlm(waypoints_image_path, clean_map_path)
        
        self._update_target_list_from_vlm()
        
    def _create_vlm_input_images(self):
        if self.no_plot:
            waypoints_image_path = '/tmp/vlm_input_waypoints.png'
        else:
            waypoints_image_path = self.filepath.replace('.png', '_vlm_input_waypoints.png')
        self.plot.plot_clean_map_with_waypoints("VLM Input - With Waypoints", waypoints_image_path, waypoints=self.original_target_list)
        
        if self.no_plot:
            clean_map_path = '/tmp/vlm_input_clean.png'
        else:
            clean_map_path = self.filepath.replace('.png', '_vlm_input_clean.png')
        self.plot.plot_clean_map_with_waypoints("VLM Input - Clean Map", clean_map_path, waypoints=None)
        
        return waypoints_image_path, clean_map_path
    
    def _encode_image(self, image_path):
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _refine_waypoints_with_vlm(self, waypoints_image_path, clean_map_path):
        base64_waypoints_image = self._encode_image(waypoints_image_path)
        base64_clean_map = self._encode_image(clean_map_path)
        
        num_waypoints = len([p for p in self.original_target_list if p != self.s_start and p != self.s_goal])
        
        prompt = f"""
You are given two images of the same maze with black walls (barriers):
1. First image: Shows the clean map with start point (blue square) and goal point (green square).
2. Second image: Shows the same map with {num_waypoints} waypoints (yellow stars) placed along a blue path. Waypoints are indexed 1..{num_waypoints}; goal is id {num_waypoints + 1}.

What is a "waypoint" and its role (read carefully):
- A waypoint (yellow star) is a navigation landmark — a coarse checkpoint placed in clearly open space that helps the robot orient its heading and follow a feasible route.
- It is NOT a precise docking coordinate. Waypoints indicate:
  - turning points (where the robot must change direction),
  - corridor transitions (entering or leaving a corridor),
  - decision junctions (where multiple passages meet).
- A valid waypoint MUST be centered in open space with visible clearance from walls. Waypoints in dead-ends, touching/near walls, or inside narrow squeezes are invalid and must be discarded.
- Because the robot travels in straight-line segments between consecutive waypoints, every such segment in the final path must be visibly open and free of contact with barriers.
- Important: The second image (with yellow stars) is only a **suggested** route — it is NOT guaranteed to be a valid robot path. You must infer safety from the barrier layout (do not assume the blue path is correct).

IMPORTANT RULES:
- This is for a physical robot. The robot cannot touch, graze, or squeeze between walls. Be conservative: if a straight segment is ambiguous or appears to touch walls, treat it as blocked.
- Do NOT create any new waypoints. Choose only from the existing numbered candidate waypoints shown in the second image.
- Do NOT output internal chain-of-thought. Output only the structured JSON described below using factual, image-tied statements.

TASK (two stages, output combined):
1) First, inspect the clean map (first image) globally and identify which corridors or directions from start toward goal are visibly open or blocked.
2) Then, using that global view, evaluate each original waypoint in order and decide whether it is essential as a navigation marker:
   - Keep a waypoint if it lies in open space and is necessary as a turning point, corridor transition, or decision marker so that start → selected_waypoint_1 → ... → goal can be realized by clearly open straight segments.
   - Discard a waypoint if it lies in a blocked, narrow, redundant, or dead-end location that would force the robot into unsafe or blocked segments.

OUTPUT (strict JSON only; nothing else):
{{
  "selected_waypoints": [ list of integer waypoint IDs to KEEP in traversal order, e.g. [2, 5] ],
  "final_reasoning": "(a) explicitly describe the overall feasible route(s) observed on the clean map before considering waypoints, and (b) explain for each chosen waypoint why it is necessary and for discarded waypoints why they were removed. Keep statements factual and tied to visible barriers/corridors."
}}

STRICT FORMATTING & CONTENT RULES:
- Output JSON only, no extra text.
- `selected_waypoints` must contain only integers between 1 and {num_waypoints}. If no original waypoint is needed, return an empty list.
- `final_reasoning` must be factual and image-referential (e.g., "left corridor open; vertical wall at x≈60 blocks direct route; waypoint 3 kept because it sits at the junction enabling a right turn into the open corridor").
- The final path implied by start → selected_waypoint_1 → ... → goal must have every consecutive straight segment clearly open (no touching/crossing walls) according to your global observations.
- Be conservative: when in doubt about clearance or collision, discard the waypoint.
"""
        
        try:
            if True:  # --- IGNORE ---
                response = self.client.chat.completions.create(
                    model=self.vlm,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_clean_map}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_waypoints_image}"}}
                        ],
                    }],
                    response_format={
                        "type": "json_schema",
                        "schema": WaypointSelection.model_json_schema(),
                    },
            )
                result = json.loads(response.choices[0].message.content)
                return result
            
        except Exception as e:
            print(f"VLM refinement failed: {e}")
            waypoint_indices = list(range(1, num_waypoints + 1))
            return {
                "selected_waypoints": waypoint_indices,
                "final_reasoning": "VLM failed, keeping all waypoints",
                "analysis": []
            }
    
    def _update_target_list_from_vlm(self):
        selected_indices = self.vlm_result.get("selected_waypoints", [])
        
        new_target_list = [self.s_start]
        
        filtered_waypoints = []
        for point in self.original_target_list:
            if point != self.s_start and point != self.s_goal:
                filtered_waypoints.append(point)
        
        for idx in selected_indices:
            if 1 <= idx <= len(filtered_waypoints):
                new_target_list.append(filtered_waypoints[idx - 1])
        
        new_target_list.append(self.s_goal)
        
        self.target_list = new_target_list
        self.i = 1
        if len(self.target_list) > 1:
            self.s_target = self.target_list[1]
        
    
    def searching(self, query, filepath='temp.png', no_plot=False):
        """
        VLM-A* searching algorithm.
        :return: Path and search metrics with VLM information.
        """
        result = super().searching(query, filepath, no_plot)
        
        result["vlm_original_waypoints"] = self.original_target_list
        result["vlm_selected_waypoints"] = self.vlm_result.get("selected_waypoints", [])
        result["vlm_reasoning"] = self.vlm_result.get("final_reasoning", "")
        
        if not self.no_plot:
            vlm_waypoints_filepath = self.filepath.replace('.png', '_vlm_waypoints.png')
            self.plot.plot_clean_map_with_waypoints("VLM-A* Waypoints", vlm_waypoints_filepath, waypoints=self.target_list)
        
        return result
