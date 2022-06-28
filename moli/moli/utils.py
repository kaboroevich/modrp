import os
import pandas as pd
import numpy as np
import random
from typing import Iterable
from functools import reduce

from sklearn.feature_selection import VarianceThreshold
import torch
from torch import Tensor
from torch.utils.data.sampler import Sampler, WeightedRandomSampler


def read_omic_dataframes(expresion_file: str, mutation_file: str, cna_file: str,
                         response_file: str):
    expression = pd.read_csv(expresion_file, sep="\t", index_col=0,
                             decimal=",").transpose()
    mutation = pd.read_csv(mutation_file, sep="\t", index_col=0,
                           decimal=".").transpose()
    cna = pd.read_csv(cna_file, sep="\t", index_col=0, decimal=".").transpose()
    cna = cna.loc[:, ~cna.columns.duplicated()]
    response = pd.read_csv(response_file, sep="\t", index_col=0, decimal=",")
    response.index = response.index.astype(str)
    return {
        'expression': expression,
        'mutation': mutation,
        'cna': cna,
        'response': response
    }


def harmonize_index(dfs: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
    shared_idx = reduce(np.intersect1d, [df.index for df in dfs])
    return shared_idx


def harmonize_columns(dfs: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
    shared_cols = reduce(np.intersect1d, [df.columns for df in dfs])
    return shared_cols


def binarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna(0)
    df[df != 0.0] = 1
    return df


def get_dataset(drug: str, training_set: str, testing_set: str,
                zenodo_dir: str) -> dict[str, dict[str, np.ndarray]]:
    cohorts = pd.read_csv(os.path.join(zenodo_dir, "Multi-OMICs.cohorts.tsv"),
                          sep="\t", index_col=0)
    # filter cohort
    cohort = cohorts[(cohorts.drug == drug) &
                     (cohorts['training set'] == training_set) &
                     (cohorts['testing set'] == testing_set)].copy()
    if cohort.shape[0] == 0:
        raise ValueError(
            f"No cohorts identified with parameters {drug}:{training_set}:"
            f"{testing_set}")
    elif cohort.shape[0] > 1:
        raise ValueError(
            f"Multiple cohorts identified with parameters {drug}:"
            f"{training_set}:{testing_set}")
    fcols = cohort.filter(regex=r'_file$', axis=1).columns
    for fcol in fcols:
        cohort[fcol] = cohort[fcol].apply(
            lambda file: os.path.join(zenodo_dir, file)) + '.gz'
    cohort = cohort.to_dict('records')[0]
    # Change to use CNA_binary as state on Zenodo 4036592
    cohort['train_CNA_file'] = cohort['train_CNA_file'].replace('CNA/',
                                                                'CNA_binary/')
    cohort['test_CNA_file'] = cohort['test_CNA_file'].replace('CNA/',
                                                              'CNA_binary/')

    # load datasets
    dset = {
        'train': read_omic_dataframes(cohort['train_exprs_file'],
                                      cohort['train_mut_file'],
                                      cohort['train_CNA_file'],
                                      cohort['train_response_file']),
        'test': read_omic_dataframes(cohort['test_exprs_file'],
                                     cohort['test_mut_file'],
                                     cohort['test_CNA_file'],
                                     cohort['test_response_file'])
    }
    # Variance selection on expression
    selector = VarianceThreshold(0.05)
    expr = dset['train']['expression']
    selector.fit_transform(expr)
    dset['train']['expression'] = expr[
        expr.columns[selector.get_support(indices=True)]]
    # harmonize data
    # filter and order samples
    for tset in ('train', 'test'):
        shared_idx = reduce(np.intersect1d,
                            [df.index for df in dset[tset].values()])
        dset[tset] = {omic: df.loc[shared_idx, :] for (omic, df) in
                      dset[tset].items()}
    # filter and order genes
    for omic in ('expression', 'mutation', 'cna'):
        shared_cols = reduce(np.intersect1d, (
            dset['train'][omic].columns, dset['test'][omic].columns))
        for tset in ('train', 'test'):
            dset[tset][omic] = dset[tset][omic].loc[:, shared_cols]
    # binary data - not required
    # for tset in ('train', 'test'):
    #     for omic in ('mutation', 'cna'):
    #        dset[tset][omic] = binarize_dataframe(dset[tset][omic])
    # reponse
    for tset in ('train', 'test'):
        dset[tset]['response'] = dset[tset]['response']['response'].apply(
            lambda x: ['R', 'S'].index(x))
    # Convert to numpy arrays
    for tset in ('train', 'test'):
        for omic in ('expression', 'mutation', 'cna', 'response'):
            dset[tset][omic] = dset[tset][omic].values

    return dset


def moli_weighted_sampler(targets: Tensor) -> Sampler:
    count = torch.unique(targets, return_counts=True)
    weight = torch.empty(targets.shape, dtype=torch.float)
    for i in range(count[0].shape[0]):
        weight[torch.where(targets == count[0][i])] = 1. / count[1][i]
    sampler = WeightedRandomSampler(weight, len(weight), replacement=True)
    return sampler


def moli_dataloader(dataset, batch_size):
    sampler = moli_weighted_sampler(dataset[:][-1])
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                         shuffle=False, sampler=sampler)
    return loader


def random_hyperparameters(seed: int = None) -> dict[str, float]:
    if seed is not None:
        random.seed(seed)
    ls_batch_size = [8, 16, 32, 64]
    ls_hlayers = [2048, 1024, 512, 256, 128, 64, 32, 16]
    ls_margin = [0.5, 1., 1.5, 2., 2.5, 3., 3.5]
    ls_lr = [0.1, 0.5, 0.01, 0.05, 0.001, 0.005,
             0.0001, 0.0005, 0.00001, 0.00005]
    ls_epoch = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
    ls_drate = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ls_wd = [0.1, 0.01, 0.001, 0.0001]
    ls_gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    hp = {
        'batch_size': random.choice(ls_batch_size),
        'exp_hl': random.choice(ls_hlayers),
        'mut_hl': random.choice(ls_hlayers),
        'cna_hl': random.choice(ls_hlayers),
        'margin': random.choice(ls_margin),
        'exp_lr': random.choice(ls_lr),
        'mut_lr': random.choice(ls_lr),
        'cna_lr': random.choice(ls_lr),
        'cls_lr': random.choice(ls_lr),
        'epochs': random.choice(ls_epoch),
        'exp_dr': random.choice(ls_drate),
        'mut_dr': random.choice(ls_drate),
        'cna_dr': random.choice(ls_drate),
        'cls_dr': random.choice(ls_drate),
        'cls_wd': random.choice(ls_wd),
        'gamma': random.choice(ls_gamma)
    }
    return hp


def optimal_hyperparameters(drug: str, training_set: str,
                            testing_set: str) -> dict[str, float]:
    hp_optima = pd.DataFrame([
        {
            'training set': 'GDSC',
            'testing set': 'PDX',
            'drug': 'Cetuximab',
            'batch_size': 16,
            'exp_hl': 32,
            'mut_hl': 16,
            'cna_hl': 256,
            'margin': 1.5,
            'exp_lr': 0.001,
            'mut_lr': 0.0001,
            'cna_lr': 5e-5,
            'cls_lr': 0.005,
            'epochs': 20,
            'exp_dr': 0.5,
            'mut_dr': 0.8,
            'cna_dr': 0.5,
            'cls_dr': 0.3,
            'cls_wd': 0.0001,
            'gamma': 0.5
        },
        {
            'training set': 'GDSC',
            'testing set': 'PDX',
            'drug': 'Erlotinib',
            'batch_size': 16,
            'exp_hl': 32,
            'mut_hl': 16,
            'cna_hl': 256,
            'margin': 1.5,
            'exp_lr': 0.001,
            'mut_lr': 0.0001,
            'cna_lr': 5e-5,
            'cls_lr': 0.005,
            'epochs': 20,
            'exp_dr': 0.5,
            'mut_dr': 0.8,
            'cna_dr': 0.5,
            'cls_dr': 0.3,
            'cls_wd': 0.0001,
            'gamma': 0.5
        },
        {
            'training set': 'GDSC',
            'testing set': 'PDX',
            'drug': 'Gemcitabine',
            'batch_size': 13,
            'exp_hl': 256,
            'mut_hl': 32,
            'cna_hl': 64,
            'margin': 1.5,
            'exp_lr': 0.05,
            'mut_lr': 1e-5,
            'cna_lr': 0.0005,
            'cls_lr': 0.001,
            'epochs': 5,
            'exp_dr': 0.4,
            'mut_dr': 0.6,
            'cna_dr': 0.3,
            'cls_dr': 0.6,
            'cls_wd': 0.01,
            'gamma': 0.3
        },
        {
            'training set': 'GDSC',
            'testing set': 'PDX',
            'drug': 'Paclitaxel',
            'batch_size': 64,
            'exp_hl': 512,
            'mut_hl': 256,
            'cna_hl': 1024,
            'margin': 0.5,
            'exp_lr': 0.0005,
            'mut_lr': 0.5,
            'cna_lr': 0.5,
            'cls_lr': 0.5,
            'epochs': 10,
            'exp_dr': 0.4,
            'mut_dr': 0.4,
            'cna_dr': 0.5,
            'cls_dr': 0.3,
            'cls_wd': 0.0001,
            'gamma': 0.6
        },
        {
            'training set': 'GDSC',
            'testing set': 'TCGA',
            'drug': 'Cisplatin',
            'batch_size': 15,
            'exp_hl': 128,
            'mut_hl': 128,
            'cna_hl': 128,
            'margin': 0.5,
            'exp_lr': 0.05,
            'mut_lr': 0.005,
            'cna_lr': 0.005,
            'cls_lr': 0.0005,
            'epochs': 20,
            'exp_dr': 0.5,
            'mut_dr': 0.6,
            'cna_dr': 0.8,
            'cls_dr': 0.6,
            'cls_wd': 0.1,
            'gamma': 0.2
        },
        {
            'training set': 'GDSC',
            'testing set': 'TCGA',
            'drug': 'Docetaxel',
            'batch_size': 8,
            'exp_hl': 16,
            'mut_hl': 16,
            'cna_hl': 16,
            'margin': 0.5,
            'exp_lr': 0.0001,
            'mut_lr': 0.0005,
            'cna_lr': 0.0005,
            'cls_lr': 0.001,
            'epochs': 10,
            'exp_dr': 0.5,
            'mut_dr': 0.5,
            'cna_dr': 0.5,
            'cls_dr': 0.5,
            'cls_wd': 0.001,
            'gamma': 0.4
        },
        {
            'training set': 'GDSC',
            'testing set': 'TCGA',
            'drug': 'Gemcitabine',
            'batch_size': 13,
            'exp_hl': 16,
            'mut_hl': 16,
            'cna_hl': 16,
            'margin': 2,
            'exp_lr': 0.001,
            'mut_lr': 0.0001,
            'cna_lr': 0.01,
            'cls_lr': 0.05,
            'epochs': 10,
            'exp_dr': 0.5,
            'mut_dr': 0.5,
            'cna_dr': 0.5,
            'cls_dr': 0.5,
            'cls_wd': 0.001,
            'gamma': 0.6
        }
    ])
    hp_optima = hp_optima[(hp_optima.drug == drug) &
                          (hp_optima['training set'] == training_set) &
                          (hp_optima['testing set'] == testing_set)].copy()
    if hp_optima.shape[0] == 0:
        raise ValueError(
            f"No cohorts identified with parameters {drug}:{training_set}:"
            f"{testing_set}")
    elif hp_optima.shape[0] > 1:
        raise ValueError(
            f"Multiple cohorts identified with parameters {drug}:"
            f"{training_set}:{testing_set}")
    return (hp_optima
            .drop(['training set', 'testing set', 'drug'], axis=1)
            .to_dict('records')[0])
