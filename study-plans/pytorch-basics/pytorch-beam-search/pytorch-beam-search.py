def beam_search(log_probs_fn, start_token, end_token, beam_width, max_len):
    """
    Returns: list of token IDs
    """
    active_beams = [([start_token],0.0)]
    completed_beams = []
    for step in range(max_len):
        if not active_beams:
            break
        all_candidates = []
        for seq, score in active_beams:
            log_probs = log_probs_fn(seq)
            for token_id, log_prob in enumerate(log_probs):
                new_seq = seq + [token_id]
                new_score = score + log_prob
                all_candidates.append((new_seq,new_score))
        all_candidates.sort(key = lambda x:x[1], reverse = True)
        active_beams = []
        for seq,score in all_candidates:
            if seq[-1] == end_token:
                completed_beams.append((seq,score))
            else:
                active_beams.append((seq,score))
                if len(active_beams) == beam_width:
                    break
    all_final_beams = completed_beams + active_beams
    if not all_final_beams:
        return [start_token]
    best_seq, best_score = max(all_final_beams,key =  lambda x: x[1])
    if best_seq[-1] == end_token:
        return best_seq[:-1]
    return best_seq