# support-data-papers
 This repository contains the support data used in the academic works I (co-)authored.

## Degraeuwe_2024
This directory contains the data used in the Degraeuwe (2024) doctoral dissertation.

### Chapter4
#### Section4_2
- ``datasetStatistics.json``: Python dictionary that presents – for each ambiguous item – the number of senses (``"n_senses"``), the number of sentences containing the ambiguous item retrieved from the SCAP corpora (``"dataset_all_size"``), the number of sentences maintained after automatic, rule-based cleaning (``"dataset_cleaned_size"``), the original size of the test set (``"test_set_orig_size"``), the size of the test set after annotation (with tagging errors and undecided instances being removed; ``"test_set_annots_size"``), the size of the rest set (``"rest_set_size"``), the distribution per sense for the test set sentences (``"label_distribution_test_set"``), and the distribution per sense for each of the three input types (``"label_distribution_supervised_base"``, ``"label_distribution_supervised_enriched"``, and ``"label_distribution_semi_supervised"``).
- ``performance.json``: Python dictionary that presents – for each ambiguous item – the number of senses (``"n_senses"``), the normalised entropy value (``"normalised_entropy"``), the performance of the most frequent sense baseline (``"performance_MFS"``), and the performance for each of the three input types (``"performance_supervised_base"``, ``"performance_supervised_enriched"``, and ``"performance_semi_supervised"``).
- ``senseInventory.json``: Python dictionary representing the sense inventory used in the word sense disambiguation study. The 74 main keys of the dictionary correspond to the 74 ambiguous items included in the study. The keys contain the lemma, part-of-speech tag, and gender of the ambiguous item separated by a vertical bar (e.g. ``"divisa|NOUN|f"``). Information on the senses is stored under the ``"senses"`` key while normalised entropy values can be retrieved by accessing the ``"normalised_entropy"`` key. The value belonging to the ``"senses"`` key is a nested dictionary containing the information per sense of the ambiguous item. The keys of this nested dictionary are strings containing the sense ID (starting at ``"1"``). The value linked to the keys is again a nested dictionary, with ``"description_ES"`` and ``"l_example_sents"`` as its keys. The ``"description_ES"`` entry contains a short description of the sense in Spanish, while the ``"l_example_sents"`` entry consists of a list of dictionaries including the prototypical example sentence(s) for that sense. Each example sentence dictionary contains the list of tokens of the sentence (accessible through the key ``"l_toks"``), the index of the ambiguous item in the sentence (accessible through the key ``"idx_ambig_item"``), and the source the sentence comes from (accessible through the key ``"source"``).
- ``testSet.txt``: tab-separated TXT file containing the individual test set annotations. The first column (``item``) contains the ambiguous item string as included in the sense inventory, the second column (``corpus_ID``) contains the ID of the SCAP corpus the sentence was taken from, the third column (``idx``) contains the index of the ambiguous item in the sentence, the fourth column (``final_label``) contains the final annotation (which corresponds to the mode of the five individual annotations), the fifth column (``individual_labels``) contains the five individual annotations, the sixth column (``target``) contains the target sentence, the seventh column (``prev``) contains the corpus sentence that precedes the target sentence, and the eighth column (``next``) contains the corpus sentence that follows the target sentence.

### Chapter5
#### Section5_1
- ``keynessExperiment_fullDataset.txt``: tab-separated TXT file containing the gold standard rankings (based on the judgements from L2 Spanish learners) and the automatic rankings (based on keyness analysis metrics) from the keyness study. Column 1 (``ID``) contains the ID of the vocabulary item, column 2 (``subset``) contains the subset of the item, column 3 (``target_word``) contains the vocabulary item (as lemma + part-of-speech separated by a vertical bar), column 4 (``normFreq_SC``) contains the item's normalised frequency per million in the source corpus, column 5 (``normFreq_RC``) contains the item's normalised frequency per million in the reference corpus, column 6 (``group``) contains the item's group, column 7 (``GS``) contains the gold standard data (ranking and underlying value, in Python dictionary format), column 8 and 9 (``baseline|rawFreq`` and ``baseline|adjFreq``) contain the baselines (ranking and underlying value, in Python dictionary format), column 10 to 17 (``LLR|freqCond1`` to ``LLR|freqCond8``) contain the data for log-likelihood ratio as the automatic keyness metric (ranking and underlying value, in Python dictionary format; for all eight frequency conditions), column 18 to 25 (``OddsRatio|freqCond1`` to ``OddsRatio|freqCond8``) contain the data for Odds Ratio as the automatic keyness metric (ranking and underlying value, in Python dictionary format; for all eight frequency conditions), and column 26 to 33 (``DKL|freqCond1`` to ``DKL|freqCond8``) contain the data for the Kullback–Leibler divergence (ranking and underlying value, in Python dictionary format; for all eight frequency conditions).

#### Section5_2
- ``BiLSTM-classifier``: this folder contains the models obtained from the tenfold cross-validation (``CV_fold1.keras`` to ``CV_fold10.keras``) as well as the Python dictionary used for the character embedding model (``d_chars_to_idxs.json``).

## References
- Degraeuwe, J. (2024). *IVESS: Intelligent Vocabulary and Example Selection for Spanish vocabulary learning* [PhD thesis]. Universiteit Gent.
