from klue_baseline.metrics.functional import *


def metric_ynat(preds, targets):
    return {'macro_f1': ynat_macro_f1(preds, targets)}


def metric_nli(preds, targets):
    return {'accuracy': klue_nli_acc(preds, targets)}


def metric_sts(preds, targets):
    return {'pearsonr': klue_sts_pearsonr(preds, targets),
            'f1': klue_sts_f1(preds, targets)}


def metric_re(probs, preds, targets):
    return {'f1': klue_re_micro_f1(preds, targets),
            'auprc': klue_re_auprc(probs, targets)}


def metric_ner(preds, targets):
    label_list = ["B-PS", "I-PS", "B-LC", "I-LC", "B-OG", "I-OG", "B-DT", "I-DT", "B-TI", "I-TI", "B-QT", "I-QT", "O"]
    return {'char_macro_f1': klue_ner_char_macro_f1(preds, targets, label_list),
            'entity_macro_f1': klue_ner_entity_macro_f1(preds, targets, label_list)}


def metric_my_ner(preds, targets):
    label_list = ["B-symptom", "I-symptom", "B-animal", "I-animal", "B-name", "I-name", "B-date", "I-date", "B-time",
                  "I-time", "B-disease", "I-disease", "B-location", "I-location", "B-hospital", "I-hospital", "O"]
    return {'char_macro_f1': klue_ner_char_macro_f1(preds, targets, label_list),
            'entity_macro_f1': klue_ner_entity_macro_f1(preds, targets, label_list)}


metrics = {
    'ynat': metric_ynat,
    'myynat': metric_nli,
    'nli': metric_nli,
    'sts': metric_sts,
    're': metric_re,
    'ner': metric_ner,
    'myner': metric_my_ner
}
