# rag_lab/evaluation/metrics.py

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

smoothie = SmoothingFunction().method4
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def evaluate_bleu(pred, ref):
    return sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)

def evaluate_rouge(pred, ref):
    scores = scorer.score(ref, pred)
    return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure

def evaluate_exact_match(pred, ref):
    return int(pred.strip() == ref.strip())


