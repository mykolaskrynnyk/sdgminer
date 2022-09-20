# standard library
import os

# sdgminer
from sdgminer.analysis import DocumentManager


def test_analysis():
    """
    Test salience and feature extraction form a .pdf document.
    """
    model_dir = os.path.join('src', 'models')
    doc_manager = DocumentManager(model_dir=model_dir, model_version='22-08-28')
    doc_manager.extract_from_file(os.path.join('tests', 'zaf_vnr_2018_main_messages.pdf'))
    doc_manager.analyse(granularity='paragraph')
    for model_type in ('multiclass', 'multilabel'):
        salience = doc_manager.get_salience(model_type=model_type)
        salience_sum = sum(salience.dictionary.values())
        assert salience_sum > 1, f'{model_type} model failed with total salience of {salience_sum:.2f}'

        top_features = doc_manager.get_top_features(model_type=model_type)
        available_features = any([len(features) > 0 for features in top_features.values()])
        assert available_features, f'{model_type} model failed with no features extracted.'