import torch
from pocket_tts import TTSModel
from safetensors.torch import load_file, save_file


def fuse_states(a, b, alpha=0.5):
    """Recursively blend two pocket-TTS state dicts (or tensors) by alpha."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape == b.shape:
            return alpha * a + (1 - alpha) * b
        return a if alpha >= 0.5 else b

    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a

    if isinstance(a, dict) and isinstance(b, dict):
        fused = {}
        for key in a.keys() | b.keys():
            if key in a and key in b:
                fused[key] = fuse_states(a[key], b[key], alpha)
            elif key in a:
                fused[key] = a[key]
            else:
                fused[key] = b[key]
        return fused

    return a if alpha >= 0.5 else b


def adjust_pitch(file_path, output_path=None, amount=0.0):
    """Shift pitch by nudging dimension 987 of the voice embedding.

    Args:
        file_path:   Path to safetensors file.
        output_path: Where to save the result (optional).
        amount:      Adjustment value (-1.0–1.0 recommended).
                     Positive = higher, negative = lower.
    """
    state = load_file(file_path)
    audio_prompt = state["audio_prompt"].clone()
    audio_prompt[:, :, 987] = audio_prompt[:, :, 987] + amount
    modified = {"audio_prompt": audio_prompt.contiguous()}
    if output_path:
        save_file(modified, output_path)
    return modified


def modify_voice(file_path, output_path=None, scale=1.0, shift=0.0):
    """Scale and/or shift all embedding values.

    Args:
        file_path:   Path to safetensors file.
        output_path: Where to save the result (optional).
        scale:       Multiply all values (>1 = more intense, <1 = subtler).
        shift:       Add a constant to all values.
    """
    state = load_file(file_path)
    audio_prompt = state["audio_prompt"]
    if scale != 1.0:
        audio_prompt = audio_prompt * scale
    if shift != 0.0:
        audio_prompt = audio_prompt + shift
    modified = {"audio_prompt": audio_prompt.contiguous()}
    if output_path:
        save_file(modified, output_path)
    return modified


def modify_dimensions(file_path, output_path=None, dim_range=None, scale=1.0, shift=0.0):
    """Scale and/or shift a specific range of embedding dimensions.

    Args:
        file_path:   Path to safetensors file.
        output_path: Where to save the result (optional).
        dim_range:   (start, end) tuple, or None to affect all dims.
        scale:       Multiply selected dimensions.
        shift:       Add to selected dimensions.
    """
    state = load_file(file_path)
    audio_prompt = state["audio_prompt"].clone()
    if dim_range:
        start, end = dim_range
        audio_prompt[:, :, start:end] = audio_prompt[:, :, start:end] * scale + shift
    else:
        audio_prompt = audio_prompt * scale + shift
    modified = {"audio_prompt": audio_prompt.contiguous()}
    if output_path:
        save_file(modified, output_path)
    return modified


def transfer_characteristic(source_file, target_file, output_path=None, dims=None, strength=1.0):
    """Copy characteristic dimensions from one voice embedding to another.

    Args:
        source_file: Voice to copy FROM.
        target_file: Voice to apply TO.
        output_path: Where to save the result (optional).
        dims:        Explicit list of dimension indices, or None to auto-select
                     the 50 dimensions that differ most between the two voices.
        strength:    Blend amount (0.0 = no change, 1.0 = full transfer).
    """
    source = load_file(source_file)["audio_prompt"]
    target = load_file(target_file)["audio_prompt"]
    result = target.clone()

    if dims is None:
        diff = (source.mean(dim=(0, 1)) - target.mean(dim=(0, 1))).abs()
        dims = diff.topk(50).indices.tolist()

    for dim in dims:
        adjustment = (source[:, :, dim].mean() - target[:, :, dim].mean()) * strength
        result[:, :, dim] = result[:, :, dim] + adjustment

    modified = {"audio_prompt": result.contiguous()}
    if output_path:
        save_file(modified, output_path)
        print(f"Transferred {len(dims)} dimensions (strength={strength}) → {output_path}")
    return modified


def amplify_differences(file_a, file_b, output_path=None, amplification=1.5):
    """Push voice A further away from voice B along every dimension.

    result = A + (A - B) * (amplification - 1)

    Args:
        file_a:        Voice to exaggerate.
        file_b:        Reference voice.
        output_path:   Where to save the result (optional).
        amplification: 1.0 = no change, >1 = more exaggerated.
    """
    prompt_a = load_file(file_a)["audio_prompt"]
    prompt_b = load_file(file_b)["audio_prompt"]
    min_len = min(prompt_a.shape[1], prompt_b.shape[1])
    amplified = prompt_a.clone()
    amplified[:, :min_len, :] = prompt_a[:, :min_len, :] + (
        prompt_a[:, :min_len, :] - prompt_b[:, :min_len, :]
    ) * (amplification - 1)
    modified = {"audio_prompt": amplified.contiguous()}
    if output_path:
        save_file(modified, output_path)
        print(f"Saved amplified voice → {output_path}")
    return modified


def fuse_safetensors(file_a, file_b, output_path=None, alpha=0.5):
    """Blend two safetensors embeddings by alpha.

    Args:
        file_a:      Path to first file (alpha weight).
        file_b:      Path to second file (1-alpha weight).
        output_path: Where to save the result (optional).
        alpha:       0.0 = 100% B, 1.0 = 100% A.
    """
    state_a = load_file(file_a)
    state_b = load_file(file_b)
    print(f"Fusing {file_a} ({alpha*100:.0f}%) + {file_b} ({(1-alpha)*100:.0f}%)...")
    fused = fuse_states(state_a, state_b, alpha)
    print("Fusion complete.")
    if output_path:
        save_file(fused, output_path)
        print(f"Saved → {output_path}")
    return fused


def fuse_voice_files(file_a, file_b, alpha=0.5):
    """Load two voice files, blend their embeddings, and return a ready-to-use model state.

    Args:
        file_a: Path to first safetensors file (alpha weight).
        file_b: Path to second safetensors file (1-alpha weight).
        alpha:  0.0 = 100% B, 1.0 = 100% A.

    Returns:
        Model state dict — pass directly to model.generate_audio().
    """
    from pocket_tts.modules.stateful_module import init_states
    model = TTSModel.load_model()

    prompt_a = load_file(file_a)["audio_prompt"].to(model.device)
    prompt_b = load_file(file_b)["audio_prompt"].to(model.device)

    min_len = min(prompt_a.shape[1], prompt_b.shape[1])
    fused_prompt = alpha * prompt_a[:, :min_len, :] + (1 - alpha) * prompt_b[:, :min_len, :]

    print("Fusing…")
    if model.flow_lm.insert_bos_before_voice:
        fused_prompt = torch.cat([model.flow_lm.bos_before_voice, fused_prompt], dim=1)
    state = init_states(model.flow_lm, batch_size=1, sequence_length=fused_prompt.shape[1])
    model._run_flow_lm_and_increment_step(model_state=state, audio_conditioning=fused_prompt)
    print("Fusion complete.")
    return state


def fuse_and_speak(file_a, file_b, text, alpha=0.5, output_path="fused_output.wav"):
    """Fuse two voices and generate speech to a WAV file.

    Args:
        file_a:       Path to first safetensors file.
        file_b:       Path to second safetensors file.
        text:         Text to synthesise.
        alpha:        Blend ratio (0.0 = 100% B, 1.0 = 100% A).
        output_path:  Output WAV path.
    """
    import soundfile as sf
    model = TTSModel.load_model()
    fused_state = fuse_voice_files(file_a, file_b, alpha)
    audio = model.generate_audio(fused_state, text)
    sf.write(output_path, audio.cpu().numpy(), model.sample_rate)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    fuse_safetensors(
        "voice_safetensors/pana.safetensors",
        "voice_safetensors/ciri.safetensors",
        output_path="voice_safetensors/pana_ciri_fused.safetensors",
        alpha=0.5,
    )
