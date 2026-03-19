PACK_SCORE_FN = None
UNPACK_SCORE_FN = None


def set_runtime_heuristics(pack_fn=None, unpack_fn=None):
    global PACK_SCORE_FN, UNPACK_SCORE_FN
    PACK_SCORE_FN = pack_fn
    UNPACK_SCORE_FN = unpack_fn


def get_pack_score_fn():
    return PACK_SCORE_FN


def get_unpack_score_fn():
    return UNPACK_SCORE_FN
