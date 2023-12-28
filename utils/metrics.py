# BLUE4
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
smoothie = SmoothingFunction().method4

def get_blue4_score(out_list, test_tgt):

    """
    out_list: [["1", "2"], ["3", "4"]]
    test_tgt: [["1", "2"], ["3", "4"]]
    """

    scores = [sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie) 
            for reference, candidate in zip(test_tgt, out_list)]
    
    return sum(scores)/len(scores)

def get_meteor_score(out_list, test_tgt):

    scores = [meteor_score([reference], candidate) 
            for reference, candidate in zip(test_tgt, out_list)]
    
    return sum(scores)/len(scores)