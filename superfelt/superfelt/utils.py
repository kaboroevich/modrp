import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from moli.utils import ods_harmonize_samples, ods_harmonize_features, \
    ods_dataframe_to_numpy, ods_label_encoder


def omic_filepaths(drug: str, train: str, test: str,
                   basedir: str) -> dict[str, str]:
    train_dir = os.path.join(basedir, train + '_without_norm')
    test_dir = os.path.join(basedir, test)
    filepath = {
        'train_exprs_file':
            os.path.join(
                test_dir,
                f"{train}_exprs.{drug}.eb_with.{test}_exprs.{drug}.tsv.gz"),
        'train_mut_file':
            os.path.join(train_dir, f"{train}.binary_mutations.tsv.gz"),
        'train_CNA_file':
            os.path.join(train_dir, f"{train}.Segment_Mean.CNA.tsv.gz"),
        'train_response_file':
            os.path.join(basedir, train, f"{train}_response.{drug}.tsv.gz"),
        'test_exprs_file':
            os.path.join(
                test_dir,
                f"{test}_exprs.{drug}.eb_with.{train}_exprs.{drug}.tsv.gz"),
        'test_mut_file':
            os.path.join(test_dir, f"{test}_mutations.{drug}.tsv.gz"),
        'test_CNA_file': os.path.join(test_dir, f"{test}_CNA.{drug}.tsv.gz"),
        'test_response_file':
            os.path.join(test_dir, f"{test}_response.{drug}.tsv.gz")
    }
    return filepath


def read_omic_dataframes(expresion_file: str, mutation_file: str, cna_file: str,
                         response_file: str):
    expression = pd.read_csv(expresion_file, sep="\t", index_col=0,
                             decimal=".").transpose()
    mutation = pd.read_csv(mutation_file, sep="\t", index_col=0,
                           decimal=".").transpose()
    cna = pd.read_csv(cna_file, sep="\t", index_col=0, decimal=".").transpose()
    cna = cna.loc[:, ~cna.columns.duplicated()]
    response = pd.read_csv(response_file, sep="\t", index_col=0, decimal=".")
    response.index = response.index.astype(str)
    return {
        'expression': expression,
        'mutation': mutation,
        'cna': cna,
        'response': response
    }


def ods_feature_select(ods: dict[str, dict[str, pd.DataFrame]]
                       ) -> dict[str, dict[str, pd.DataFrame]]:
    # expression
    vts = VarianceThreshold(0.05*20)
    exp = ods['train']['expression']
    vts.fit_transform(exp)
    ods['train']['expression'] = exp[exp.columns[vts.get_support(indices=True)]]
    # mutation
    vts = VarianceThreshold(0.00001 * 15)
    mut = ods['train']['mutation']
    vts.fit_transform(mut)
    ods['train']['mutation'] = mut[mut.columns[vts.get_support(indices=True)]]
    # CNA
    vts = VarianceThreshold(0.01 * 20)
    cna = ods['train']['cna']
    vts.fit_transform(cna)
    ods['train']['cna'] = cna[cna.columns[vts.get_support(indices=True)]]
    return ods


def ods_binarize(ods: dict[str, dict[str, pd.DataFrame]], bomics: list,
                 ) -> dict[str, dict[str, pd.DataFrame]]:
    for tset in ods.keys():
        for omic in ods[tset].keys():
            if omic in bomics:
                ods[tset][omic][ods[tset][omic] != 0.] = 1.
    return ods


def get_dataset(drug: str, train_set: str, test_set: str, zenodo_dir: str,
                ic50: bool = False, feature_selection: bool = True
                ) -> dict[str, dict[str, np.ndarray]]:
    filepath = omic_filepaths(drug, train_set, test_set, zenodo_dir)
    # load datasets
    dfset = {
        'train': read_omic_dataframes(filepath['train_exprs_file'],
                                      filepath['train_mut_file'],
                                      filepath['train_CNA_file'],
                                      filepath['train_response_file']),
        'test':  read_omic_dataframes(filepath['test_exprs_file'],
                                      filepath['test_mut_file'],
                                      filepath['test_CNA_file'],
                                      filepath['test_response_file'])}
    #
    if feature_selection:
        dfset = ods_feature_select(dfset)
    dfset = ods_binarize(dfset, ['mutation', 'cna'])
    dfset = ods_harmonize_samples(dfset)
    dfset = ods_harmonize_features(dfset)
    dfset = ods_label_encoder(dfset, ic50)
    npset = ods_dataframe_to_numpy(dfset)
    return npset


def hyperparameter_tuning_sets():
    # taken from
    # https://github.com/DMCB-GIST/Super.FELT/blob/main/Super_FELT_main.py
    hp_base = {
        'margin': 1.,
        'enc_lr': 0.01,
        'cls_lr': 0.01,
        'batch_size': 55,
        'exp_hl': 256,
        'mut_hl': 32,
        'cna_hl': 64,
        'exp_ep': 10,
        'mut_ep': 3,
        'cna_ep': 5,
        'cls_ep': 5
    }
    hp_base['epochs'] = np.max([hp_base['exp_ep'], hp_base['mut_ep'],
                                hp_base['cna_ep']]) + hp_base['cls_ep']
    hp_tune = [
        {'enc_dr': 0.1, 'enc_wd': 0.00, 'cls_dr': 0.1, 'cls_wd': 0.00},
        {'enc_dr': 0.3, 'enc_wd': 0.01, 'cls_dr': 0.3, 'cls_wd': 0.01},
        {'enc_dr': 0.3, 'enc_wd': 0.05, 'cls_dr': 0.3, 'cls_wd': 0.01},
        {'enc_dr': 0.5, 'enc_wd': 0.01, 'cls_dr': 0.5, 'cls_wd': 0.01},
        {'enc_dr': 0.5, 'enc_wd': 0.10, 'cls_dr': 0.7, 'cls_wd': 0.15},
        {'enc_dr': 0.3, 'enc_wd': 0.01, 'cls_dr': 0.5, 'cls_wd': 0.01},
        {'enc_dr': 0.4, 'enc_wd': 0.01, 'cls_dr': 0.4, 'cls_wd': 0.01},
        {'enc_dr': 0.5, 'enc_wd': 0.10, 'cls_dr': 0.5, 'cls_wd': 0.1}
    ]
    hp_set = [hp_base | x for x in hp_tune]
    return hp_set
