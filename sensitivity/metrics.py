from scipy.special import comb
import math

def jaccard_similarity(a, b):
    """list a, list b, return jaccard similarity score"""

    union = set(a) | set(b)
    insect = set.intersection(set(a), set(b))

    print('insect', insect)
    print('insect counts', len(insect))

    print('union', union)
    print('union counts', len(union))
    return len(insect)/len(union)


def mean_pairwise_Jaccard_Similarity (topic_lists, k):
    """compute mean pairwise Jaccard similarity between the topic descriptors (top ranking terms)
    " High similarity values indicate increased topic dependency
    """
    sum = 0
    for j in range(1, k):
        for i in range(0, j):

            topic_a = topic_lists[i]
            topic_b = topic_lists[j]
            score = jaccard_similarity(topic_a, topic_b)
            sum += score
            print('jaccard_similarity between topic {} and {} is {}'.format(j, i, score))

    print(comb(k, 2))
    sum = sum/comb(k, 2)
    return sum


def UMass(w_a, w_b, docs, e=10e-12):
    """
    umass metric based on document co-occurrence
    :param w_a: word a
    :param w_b: word b
    :param e:
    :param docs: documents
    :return:umass score
    """
    cocur_a_b = docs.search_cocur([w_a, w_b])
    cocur_b = docs.search_cocur([w_b])

    score = math.log((cocur_a_b+e)/cocur_b, 10)
    return score

def coherence_by_UMass(topic_words, docs):
    coherence = 0
    n_top_words = len(topic_words)
    for j in range(1, n_top_words):
        for i in range(0, j):
            w_a = topic_words[i]
            w_b = topic_words[j]
            coherence += UMass(w_a, w_b, docs)
    return coherence

def avg_coherence_by_UMass(top_words_topics, docs, k):
    """
    compute the coherence for a specific topic
    :param topic_words: lists of topics represented by top ranked terms/words
    :param docs: class Document
    :return: average of coherence for all topics
    """
    sum = 0
    for topic_idx, topic_words in enumerate(top_words_topics):
        score = coherence_by_UMass(topic_words, docs)
        print("topic {} coherence score is {}".format(topic_idx, score))
        sum += score
    return sum/k







# a = [1, 3, 4]
# b = [2, 3, 1]
# score = jaccard_similarity(a, b)
#
# topic_lists =[[2,3,10],[3,1,10]]
#
# print(score)
# print("mpj", compute_mpj(topic_lists,2))