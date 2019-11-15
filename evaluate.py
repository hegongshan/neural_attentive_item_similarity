import math

def get_hit_ratio(ranklist, gtItem):
    if gtItem in ranklist:
        return 1
    return 0


def get_NDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0
