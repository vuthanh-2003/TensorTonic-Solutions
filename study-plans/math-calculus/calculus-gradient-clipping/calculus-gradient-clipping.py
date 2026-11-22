import numpy as np

def gradient_clipping(gradients: list, max_norm: float) -> dict:
    """
    Returns clipped gradients, norms, and clipping checks.
    """
    flat_grads = (
      np.concatenate(
          [np.asarray(g, dtype=np.float64).ravel() for g in gradients]
      )
      if len(gradients) > 0
      else np.array([], dtype=np.float64)
  )
    original_norm = float(np.linalg.norm(flat_grads))
    was_clipped = bool(original_norm > max_norm)
    if was_clipped and original_norm > 0.0:
        scale = float(max_norm) / original_norm
        clipped_gradients = [
        (np.asarray(g, dtype=np.float64) * scale).tolist() for g in gradients
    ]
    else:
        scale = 1.0
        clipped_gradients = [
        np.asarray(g, dtype=np.float64).tolist() for g in gradients
    ]
    clipped_flat = (
      np.concatenate(
          [np.asarray(g, dtype=np.float64).ravel() for g in clipped_gradients]
      )
    if len(clipped_gradients) > 0
    else np.array([], dtype=np.float64)
  )
    clipped_norm = float(np.linalg.norm(clipped_flat))
    if original_norm > 0.0 and clipped_norm > 0.0:
        cosine_similarity = float(
        np.dot(flat_grads, clipped_flat) / (original_norm * clipped_norm)
    )
    else:
        cosine_similarity = 1.0
    return {
      "clipped_gradients": clipped_gradients,
      "original_global_norm": original_norm,
      "clipped_global_norm": clipped_norm,
      "was_clipped": was_clipped,
      "direction_preserved": bool(abs(cosine_similarity - 1.0) <= 1e-6),
  }
    
