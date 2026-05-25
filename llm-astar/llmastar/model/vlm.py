import json
import base64
import os
from typing import List, Tuple
from pydantic import BaseModel
from together import Together
from dotenv import load_dotenv

# Load environment variables from .env file in the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(env_path)

# Initialize client
client = Together(api_key=os.getenv('TOGETHER_API_KEY'))


from typing import List
from pydantic import BaseModel, Field

class WaypointSelection(BaseModel):
    selected_waypoints: List[int] = Field(
        description="The list of original waypoints to keep (e.g., [1, 4, 7])."
    )
    new_waypoints: List[Tuple[int, int]] = Field(
        description="New waypoints added as [x, y] coordinates to avoid barriers."
    )
    add_waypoints: List[int] = Field(
        description="Indices in the path sequence where new waypoints should be inserted."
    )
    final_path: List[str] = Field(
        description='The ordered path from start to goal, e.g., ["start", "1", "new_1", "4", "goal"].'
    )
    final_reasoning: str = Field(
        description="Short summary explaining why these waypoints (and new ones) were chosen for the robot."
    )

# -------------------- Helpers -------------------- #
def encode_image(image_path: str) -> str:
    """Encode local image into base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# -------------------- Main -------------------- #
import os
def main():
    # imagePath = "/Users/apple/Desktop/All/MSOLAB/LLM Robotics Planning/Code-llm/vlm_standard_vlm_input.png"
    imagePath = os.path.join(os.getcwd(), 'vlm_standard_vlm_input.png')
    base64_image = encode_image(imagePath)

    prompt = """
You are given an image of a maze with black walls (barriers).
The map shows:
- A start point (blue square) and a goal point (green square).
- N original waypoints (yellow stars) placed along a blue path, indexed 1..N; goal is id N+1.

PRIMARY OBJECTIVE:
Produce a safe robot route from "start" to "goal" that contains only:
  - necessary original waypoint IDs (e.g., "1","4",...), 
  - optional new waypoints ("new_1","new_2",...) with integer coordinates [x,y],
  - and "start" / "goal".

MANDATORY CHECKS (READ CAREFULLY):
1. This is a robot path: the robot cannot squeeze through, touch, or graze walls. Safety is absolute — if a straight-line segment would cross, touch, or pass too close to any wall it is NOT allowed.
2. For every consecutive pair of nodes in your proposed final_path (e.g., "start" -> "1", "1" -> "new_1", "new_1" -> "4", ...), you MUST examine that straight segment very carefully against the image and ensure it is clearly open and unobstructed. Do this *for the whole route*, not just locally.
3. If any consecutive segment in your candidate path is blocked or too narrow, you MUST fix the path by either:
   - removing the problematic original waypoint(s) (do not keep invalid points), and/or
   - inserting one or more NEW waypoint(s) at integer image coordinates [x,y] placed clearly in open space to create only safe straight segments.
4. Prefer the minimal number of kept original waypoints and the minimal number of new waypoints consistent with safety. Safety overrides minimality.
5. NEW waypoints are indicators/checkpoints only — place them in obvious open corridors and only to ensure all consecutive straight segments are safe.
6. Do NOT output any per-segment analysis, no internal reasoning, no intermediate checks. Do NOT output uncertainty. Only output the final JSON described below.

OUTPUT (STRICT JSON ONLY):
{
  "selected_waypoints": [ list of integer original IDs kept, e.g. [2,4] ],
  "new_waypoints": [ [x,y], [x,y], ... ],        // integer pixel coordinates for new checkpoints
  "add_waypoints": [ list of insertion indices (1-based) ], // insertion positions: 1 = immediately after "start"
  "final_path": [ "start", "1", "new_1", "4", "goal" ],
  "final_reasoning": "Two short sentences. First sentence MUST be exactly 'Original path is VALID.' or 'Original path is INVALID.' The second sentence states which original waypoints were kept (if any) and why new waypoint(s) were added (if any)."
}

STRICT FORMATTING RULES:
- Output JSON only, nothing else.
- final_path must be a valid sequence from "start" to "goal", and for every consecutive pair listed there must be a clearly open straight segment in the image (no touching/crossing/narrow scrape against walls).
- If no subset of original waypoints can form a safe path, produce new_waypoints that yield a safe route and explain this in final_reasoning.
- If you keep an original waypoint, it must be because all straight segments connecting it (to its predecessor and successor in the final_path) are clearly free of barriers.

SAMPLE final_reasoning (format requirement):
- Correct example: "Original path is INVALID. Waypoints 2 and 4 are kept because the straight segments between start→2, 2→new_1, new_1→4 and 4→goal are clearly open; new_1 at [45,23] was added to bypass a blocked corridor."
- final_reasoning must be 1–2 sentences only, beginning exactly with "Original path is VALID." or "Original path is INVALID."

END.
    """

    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ],
        }],
        response_format={
            "type": "json_schema",
            "schema": WaypointSelection.model_json_schema(),
        },
    )

    output = json.loads(response.choices[0].message.content)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output

if __name__ == "__main__":
    main()
