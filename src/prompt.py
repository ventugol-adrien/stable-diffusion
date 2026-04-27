def process_prompt(prompt: str, negative_prompt: str = None, model: str = "juggernaut"):
    """
    Process the user input prompt and return positive and negative prompts.
    For simplicity, this example assumes the entire input is a positive prompt.
    You can enhance this function to parse for negative prompts or special tokens.
    """
    prompt = prompt.strip()
    if "pony" in model.lower():
        positive_prompt = f"score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, {prompt}"
        negative_prompt = f"{negative_prompt or "score_6, score_5, score_4, ugly, deformed, noisy, low_quality, bad_hands, bad_feet, bad_eyes, bad_lighting, bad_anatomy"}"
    else:
        positive_prompt = (
            f"{prompt}, best quality, masterpiece, 8k, incredible detail, sharp focus"
        )
        negative_prompt = f"{negative_prompt or 'low quality, worst quality, blurry, deformed, bad anatomy, disfigured, poorly drawn, extra limbs, mutated hands, poorly drawn hands, missing fingers, extra fingers, mutated fingers, long neck'}"

    return positive_prompt, negative_prompt
