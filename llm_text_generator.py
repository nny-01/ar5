import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMTextGenerator:
    """LLM-based text prompt generator for RGBT AR.

    Supports two modes:
    1. Shared mode (legacy): all classes share one prompt set
    2. Per-class mode (new): each class gets its own description samples,
       enabling class-based compensation in the new AR design.

    Per-class mode generates descriptions like:
        car: ["a car on the road", "a small parked car", ...]
        person: ["a walking pedestrian", "a person standing", ...]
    and produces a class_map tensor: [0,0,0,..., 1,1,1,...]
    """

    def __init__(
        self,
        api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0.7,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.client = None

        # cache shared prompts so repeated class calls return the same shared set
        self._shared_relational = None
        self._shared_rgb = None
        self._shared_ir = None

    def _init_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for LLMTextGenerator. "
                "Install it with: pip install openai"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _call_api(self, system_prompt, user_prompt):
        if self.client is None:
            self._init_client()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def _parse_list_response(self, response_text):
        try:
            result = json.loads(response_text)
            if isinstance(result, list):
                return [s.strip() for s in result if s.strip()]
        except (json.JSONDecodeError, TypeError):
            pass

        if "[" in response_text and "]" in response_text:
            start = response_text.index("[")
            end = response_text.rindex("]") + 1
            try:
                result = json.loads(response_text[start:end])
                if isinstance(result, list):
                    return [s.strip() for s in result if s.strip()]
            except (json.JSONDecodeError, TypeError):
                pass

        lines = response_text.strip().split("\n")
        prompts = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for prefix_pattern in [".", ")", ":", "-", "*"]:
                parts = line.split(prefix_pattern, 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    line = parts[1].strip()
                    break
            if line.startswith(("- ", "* ")):
                line = line[2:].strip()
            if (line.startswith('"') and line.endswith('"')) or (
                line.startswith("'") and line.endswith("'")
            ):
                line = line[1:-1]
            if line:
                prompts.append(line)
        return prompts

    def _dedup(self, prompts):
        seen = set()
        out = []
        for p in prompts:
            key = "".join(ch.lower() for ch in p if ch.isalnum() or ch.isspace()).strip()
            if key and key not in seen:
                seen.add(key)
                out.append(p.strip())
        return out

    def generate_relational(self, class_name, num_prompts=5, scene_context="RGBT surveillance"):
        """Signature unchanged.
        Now generates SHARED relational prompts and returns same set for every class.
        """
        if self._shared_relational is not None and len(self._shared_relational) >= num_prompts:
            return self._shared_relational[:num_prompts]

        system_prompt = (
            "You are an expert in object detection and computer vision. "
            "Generate short, concise SHARED text prompts describing common spatial "
            "and contextual relationships in RGBT surveillance scenes. "
            "IMPORTANT RULES: "
            "1. Do NOT mention any specific object class names. "
            "2. Each prompt should be 3-8 words. "
            "3. Focus on common spatial/contextual patterns in traffic or surveillance scenes. "
            "4. Return ONLY a JSON array of strings, no other text."
        )
        user_prompt = (
            f"For a {scene_context} scene, generate {num_prompts} SHARED relational prompts. "
            f"These prompts will be shared by all object classes. "
            f'Examples: "near roadside structure", "beside lane boundary", '
            f'"under street lighting", "along traffic flow", "at road edge". '
            f"Do not mention class names like person, car, bus, truck, lamp, motorcycle. "
            f"Return a JSON array of {num_prompts} strings."
        )

        response = self._call_api(system_prompt, user_prompt)
        prompts = self._dedup(self._parse_list_response(response))
        self._shared_relational = prompts[:num_prompts]
        logger.info(f"Generated {len(self._shared_relational)} shared relational prompts")
        return self._shared_relational

    def generate_rgb_descriptive(self, class_name, num_prompts=5):
        """Signature unchanged.
        Now generates SHARED RGB descriptive prompts and returns same set for every class.
        """
        if self._shared_rgb is not None and len(self._shared_rgb) >= num_prompts:
            return self._shared_rgb[:num_prompts]

        system_prompt = (
            "You are an expert in object detection and computer vision. "
            "Generate short SHARED text prompts describing generic visible-light "
            "(RGB) appearance patterns useful for detection. "
            "IMPORTANT RULES: "
            "1. Do NOT mention any specific object class name. "
            "2. Focus ONLY on RGB-visible features: shape, outline, structure, "
            "surface, visible pattern, edge contrast. "
            "3. Do NOT mention thermal, infrared, heat, or temperature-related features. "
            "4. Each prompt should be 3-10 words. "
            "5. Return ONLY a JSON array of strings."
        )
        user_prompt = (
            f"Generate {num_prompts} SHARED RGB descriptive prompts for all classes. "
            f'These prompts should describe generic visible patterns, such as: '
            f'"clear structural outline", "compact rigid shape", '
            f'"elongated visible body", "vertical narrow structure", '
            f'"strong edge contrast". '
            f"Do not mention class names. "
            f"Return a JSON array of {num_prompts} strings."
        )

        response = self._call_api(system_prompt, user_prompt)
        prompts = self._dedup(self._parse_list_response(response))
        self._shared_rgb = prompts[:num_prompts]
        logger.info(f"Generated {len(self._shared_rgb)} shared RGB descriptive prompts")
        return self._shared_rgb

    def generate_ir_descriptive(self, class_name, num_prompts=5):
        """Signature unchanged.
        Now generates SHARED IR descriptive prompts and returns same set for every class.
        """
        if self._shared_ir is not None and len(self._shared_ir) >= num_prompts:
            return self._shared_ir[:num_prompts]

        system_prompt = (
            "You are an expert in object detection and computer vision. "
            "Generate short SHARED text prompts describing generic thermal/infrared "
            "appearance patterns useful for detection. "
            "IMPORTANT RULES: "
            "1. Do NOT mention any specific object class name. "
            "2. Focus ONLY on thermal features: bright contour, warm region, "
            "heat contrast, thermal silhouette, localized hot structure. "
            "3. Do NOT mention RGB color, texture, clothing color, or visible-light paint. "
            "4. Each prompt should be 3-10 words. "
            "5. Return ONLY a JSON array of strings."
        )
        user_prompt = (
            f"Generate {num_prompts} SHARED IR descriptive prompts for all classes. "
            f'These prompts should describe generic thermal patterns, such as: '
            f'"bright thermal contour", "localized warm region", '
            f'"compact warm silhouette", "stable heat contrast", '
            f'"cool background separation". '
            f"Do not mention class names. "
            f"Return a JSON array of {num_prompts} strings."
        )

        response = self._call_api(system_prompt, user_prompt)
        prompts = self._dedup(self._parse_list_response(response))
        self._shared_ir = prompts[:num_prompts]
        logger.info(f"Generated {len(self._shared_ir)} shared IR descriptive prompts")
        return self._shared_ir

    # ================================================================
    # Per-class description generation (for class-based compensation)
    # ================================================================
    def generate_class_rgb_descriptive(self, class_name, num_prompts=5):
        """Generate RGB descriptive prompts specific to a single class.

        Unlike generate_rgb_descriptive (shared), this produces descriptions
        that are specific to the given class, e.g. for 'car':
            ["a car on the road", "a small parked car", ...]
        """
        system_prompt = (
            "You are an expert in object detection and computer vision. "
            "Generate short text prompts describing the visible-light (RGB) "
            "appearance of a specific object class for detection. "
            "IMPORTANT RULES: "
            "1. Each prompt MUST describe the given class from different viewpoints, "
            "sizes, poses, or contexts. "
            "2. Focus ONLY on RGB-visible features: shape, color, size, posture, "
            "surface texture, structural outline. "
            "3. Do NOT mention thermal, infrared, heat, or temperature features. "
            "4. Each prompt should be 3-12 words. "
            "5. Return ONLY a JSON array of strings, no other text."
        )
        user_prompt = (
            f"Generate {num_prompts} RGB descriptive prompts for the class '{class_name}'. "
            f"Each prompt should describe a different visual appearance of '{class_name}'. "
            f"Examples for 'car': "
            f'["a car driving on the road", "a small parked car", '
            f'"a car with bright headlights", "a large SUV on highway", '
            f'"a car partially occluded by tree"]. '
            f"Now generate {num_prompts} for '{class_name}'. "
            f"Return a JSON array of {num_prompts} strings."
        )

        response = self._call_api(system_prompt, user_prompt)
        prompts = self._dedup(self._parse_list_response(response))
        result = prompts[:num_prompts]
        logger.info(f"Generated {len(result)} per-class RGB descriptive prompts for '{class_name}'")
        return result

    def generate_class_ir_descriptive(self, class_name, num_prompts=5):
        """Generate IR descriptive prompts specific to a single class.

        Unlike generate_ir_descriptive (shared), this produces descriptions
        that are specific to the given class, e.g. for 'person':
            ["a warm human silhouette", "a bright thermal figure walking", ...]
        """
        system_prompt = (
            "You are an expert in object detection and computer vision. "
            "Generate short text prompts describing the thermal/infrared "
            "appearance of a specific object class for detection. "
            "IMPORTANT RULES: "
            "1. Each prompt MUST describe the given class from different thermal "
            "perspectives: heat signature, thermal contrast, silhouette shape. "
            "2. Focus ONLY on thermal/infrared features. "
            "3. Do NOT mention RGB color, texture, or visible-light features. "
            "4. Each prompt should be 3-12 words. "
            "5. Return ONLY a JSON array of strings, no other text."
        )
        user_prompt = (
            f"Generate {num_prompts} thermal/infrared descriptive prompts for the class '{class_name}'. "
            f"Each prompt should describe a different thermal appearance of '{class_name}'. "
            f"Examples for 'person': "
            f'["a warm human silhouette", "a bright thermal figure walking", '
            f'"a person with strong heat signature", "a dim thermal pedestrian at distance", '
            f'"a warm body against cool background"]. '
            f"Now generate {num_prompts} for '{class_name}'. "
            f"Return a JSON array of {num_prompts} strings."
        )

        response = self._call_api(system_prompt, user_prompt)
        prompts = self._dedup(self._parse_list_response(response))
        result = prompts[:num_prompts]
        logger.info(f"Generated {len(result)} per-class IR descriptive prompts for '{class_name}'")
        return result

    def generate_perclass_for_dataset(self, class_names, num_descriptive=5,
                                      scene_context="RGBT surveillance"):
        """Generate per-class RGB and IR descriptions for all classes.

        Returns:
            dict with keys:
                'per_class': {class_name: {'desc_rgb': [...], 'desc_ir': [...], 'class_idx': int}}
                'all_desc_rgb': list of all RGB descriptions concatenated
                'all_desc_ir': list of all IR descriptions concatenated
                'desc_rgb_class_map': list of class indices for each RGB description
                'desc_ir_class_map': list of class indices for each IR description
        """
        per_class = {}
        all_desc_rgb = []
        all_desc_ir = []
        desc_rgb_class_map = []
        desc_ir_class_map = []

        for idx, name in enumerate(class_names):
            logger.info(f"Generating per-class descriptions for '{name}' (class {idx})...")
            rgb_descs = self.generate_class_rgb_descriptive(name, num_descriptive)
            ir_descs = self.generate_class_ir_descriptive(name, num_descriptive)

            per_class[name] = {
                'class_idx': idx,
                'desc_rgb': rgb_descs,
                'desc_ir': ir_descs,
            }

            all_desc_rgb.extend(rgb_descs)
            all_desc_ir.extend(ir_descs)
            desc_rgb_class_map.extend([idx] * len(rgb_descs))
            desc_ir_class_map.extend([idx] * len(ir_descs))

        logger.info(
            f"Per-class generation complete: {len(class_names)} classes, "
            f"{len(all_desc_rgb)} RGB descs, {len(all_desc_ir)} IR descs"
        )

        return {
            'per_class': per_class,
            'all_desc_rgb': all_desc_rgb,
            'all_desc_ir': all_desc_ir,
            'desc_rgb_class_map': desc_rgb_class_map,
            'desc_ir_class_map': desc_ir_class_map,
        }

    # ================================================================
    # Legacy shared-mode interfaces (kept for compatibility)
    # ================================================================
    def generate(self, class_name, num_relational=5, num_descriptive=5,
                 scene_context="RGBT surveillance"):
        """Legacy interface. Returns shared prompts."""
        relational = self.generate_relational(class_name, num_relational, scene_context)
        desc_rgb = self.generate_rgb_descriptive(class_name, num_descriptive)
        desc_ir = self.generate_ir_descriptive(class_name, num_descriptive)
        return {
            "class_name": class_name,
            "relational": relational,
            "desc_rgb": desc_rgb,
            "desc_ir": desc_ir,
            "all_prompts": relational + desc_rgb + desc_ir,
        }

    def generate_for_dataset(self, class_names, num_relational=5, num_descriptive=5,
                             scene_context="RGBT surveillance"):
        """Legacy interface. Returns shared prompts for each class."""
        results = {}
        for name in class_names:
            logger.info(f"Generating prompts for class '{name}'...")
            results[name] = self.generate(name, num_relational, num_descriptive,
                                          scene_context)
        return results

    def save_prompts(self, prompts, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        logger.info(f"Prompts saved to {save_path}")

    @staticmethod
    def load_prompts(load_path):
        with open(load_path, encoding="utf-8") as f:
            return json.load(f)