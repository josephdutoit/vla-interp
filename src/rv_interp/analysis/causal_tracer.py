import torch
import re
from qwen_vl_utils import process_vision_info
from rv_train.model import QwenVLActor

VISION_END_TAG = '<|vision_end|>'


class CausalTracer:
    def __init__(self, model: QwenVLActor):
        self.model = model.model 
        self.processor = model.processor
        self.device = model.device
        self.stored_activations = None
        self.target_indices = None 

    def _save_hook(self, mod, inp, out):
        data = out[0] if isinstance(out, tuple) else out
        self.stored_activations = data.detach().cpu()

    def _patch_hook(self, mod, inp, out):
        clean_data = self.stored_activations.to(self.device)
        corrupted_data = out[0] if isinstance(out, tuple) else out
        patched_data = corrupted_data.clone()
        if self.target_indices is not None:
            patched_data[:, self.target_indices, :] = clean_data[:, self.target_indices, :]
        
        if isinstance(out, tuple):
            return (patched_data,) + out[1:]
        return patched_data

    def _get_action_logit_indices(self, input_ids, k_action_steps = 0, horizon=8):
        """Helper to find the indices of the 56 tokens representing actions."""
        action_token_indices = []
        k_action_steps_idx = 0
        k_actions_to_find = 7 * (horizon - k_action_steps)
        num_actions_found = 0
        for i in range(len(input_ids) - 2, 0, -1):
            token_str = self.processor.tokenizer.decode([input_ids[i]])
            if re.sub(r'\s+', '', token_str).isnumeric() or token_str == " ":
                action_token_indices.append(i)
                if ' ' in token_str:
                    num_actions_found += 1
            if num_actions_found >= k_actions_to_find and k_action_steps_idx == 0:
                k_action_steps_idx = len(action_token_indices)
            if token_str == '\n':
                num_actions_found += 1
            if num_actions_found == 7 * horizon:
                break

        action_token_indices.reverse()
        k_action_steps_idx = len(action_token_indices) - k_action_steps_idx

        return [idx - 1 for idx in action_token_indices], k_action_steps_idx - 1

    def _get_instruction_token_indices(self, input_ids, string_indices):
        vision_end_id = self.processor.tokenizer.convert_tokens_to_ids(VISION_END_TAG)
        ids_list = input_ids.tolist()
        vision_boundary_idx = len(ids_list) - 1 - ids_list[::-1].index(vision_end_id)

        target_indices = []
        current_pos = 0
        for i in range(vision_boundary_idx + 1, len(ids_list)):
            token_str = self.processor.tokenizer.decode([ids_list[i]])
            token_len = len(token_str)
            for c_start, c_end in string_indices:
                if max(current_pos, c_start) < min(current_pos + token_len, c_end):
                    if ids_list[i] > 100: target_indices.append(i)
            current_pos += token_len
        return target_indices

    def _compute_entropy_and_confidence(self, logits):
        log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
        probs = torch.exp(log_probs)
        token_entropies = -torch.sum(probs * log_probs, dim=-1)
        max_probs = torch.max(probs, dim=-1).values
        return token_entropies, max_probs

    @torch.no_grad()
    def trace_frame(
        self, 
        data, 
        layer_range: range, 
        noise_std=0.5,
        trace_target="language"
    ):
        text = self.processor.apply_chat_template(data["messages"], tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(data["messages"])
        inputs = self.processor(text=[text], images=image_inputs, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]
        
        
        if trace_target == "language":
            self.target_indices = self._get_instruction_token_indices(input_ids, data["corrupted_idxs"])
            logit_indices, _ = self._get_action_logit_indices(input_ids)
        elif trace_target == "action":
            logit_indices, corrupt_idx = self._get_action_logit_indices(input_ids, k_action_steps=1)
            self.target_indices = logit_indices[:corrupt_idx]
            logit_indices = logit_indices[corrupt_idx:]


        text_embeds = self.model.model.language_model.embed_tokens(inputs['input_ids'])
        clean_embeds = text_embeds.clone()
        corrupted_embeds = clean_embeds.clone()
        noise = torch.randn_like(corrupted_embeds[:, self.target_indices, :]) * noise_std
        corrupted_embeds[:, self.target_indices, :] += noise

        clean_out = self.model(inputs_embeds=clean_embeds, images=image_inputs)
        corr_out = self.model(inputs_embeds=corrupted_embeds, images=image_inputs)
        
        clean_traj_logits = clean_out.logits[0, logit_indices, :] 
        corr_traj_logits = corr_out.logits[0, logit_indices, :]

        clean_token_entropies, clean_max_probs = self._compute_entropy_and_confidence(clean_traj_logits)
        corrupted_token_entropies, corrupted_max_probs = self._compute_entropy_and_confidence(corr_traj_logits)

        baseline_dist = torch.dist(clean_traj_logits, corr_traj_logits).item()

        recovery_curve = []
        for layer_idx in layer_range:
            target_layer = self.model.model.language_model.layers[layer_idx]
            
            h_save = target_layer.register_forward_hook(self._save_hook)
            self.model(inputs_embeds=clean_embeds, images=image_inputs)
            h_save.remove()

            h_patch = target_layer.register_forward_hook(self._patch_hook)
            patched_out = self.model(inputs_embeds=corrupted_embeds, images=image_inputs)
            patched_traj_logits = patched_out.logits[0, logit_indices, :]
            h_patch.remove()

            patch_dist = torch.dist(patched_traj_logits, clean_traj_logits).item()
            recovery = (baseline_dist - patch_dist) / (baseline_dist + 1e-9)
            recovery_curve.append(max(0.0, min(1.0, recovery)))
            
            self.stored_activations = None 

        return {
            "recovery_curve": recovery_curve,
            "clean_token_entropies": clean_token_entropies.cpu().tolist(),
            "corrupted_token_entropies": corrupted_token_entropies.cpu().tolist(),
            "clean_max_probs": clean_max_probs.cpu().tolist(),
            "corrupted_max_probs": corrupted_max_probs.cpu().tolist(),
            "baseline_dist": baseline_dist,
            "metadata": {
                "instruction": data["messages"][1]["content"][1]["text"],
                "timestamp": float(data.get("timestamp", 0)),
                "frame_idx": int(data.get("frame_index", 0)),
                "episode_idx": int(data.get("episode_index", 0))
            }
        }