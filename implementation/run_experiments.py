import utils

import pandas as pd
import time
import math
import re
import json
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# from matplotlib import pyplot as plt
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# import sklearn.utils


df_questions = pd.read_pickle("../data/datasets/duplicate_questions_exp_prep.pkl")
df_pairs = pd.read_pickle("../data/datasets/duplicate_pairs_exp_prep.pkl")

# Create the specific dataset
set_positive_pairs = utils.create_positive_pair_set(df_pairs)
df_pairs_training = df_pairs.head(8000)
df_pairs_others = df_pairs.tail(2000)
df_pairs_validation = df_pairs_others.head(1000)
df_pairs_testing = df_pairs_others.tail(1000)
questions_set_training = utils.create_set_questions_df(df_pairs_training)
questions_set_validating = utils.create_set_questions_df(df_pairs_validation)
questions_set_testing = utils.create_set_questions_df(df_pairs_testing)

questions_set_learning = questions_set_training.union(questions_set_validating)
questions_set_testing_use = questions_set_testing.difference(questions_set_training.union(questions_set_validating))

###################These prints might be useful
# print(len(questions_set_validating), "print the dimension of the validation set")
# print(len(questions_set_validating.difference(questions_set_training)),
#       "print the dimenstion of the questions from vlidation but not in training")

# print(len(questions_set_testing.union(questions_set_validating.union(questions_set_training))),
#       "print the number of questions which will be used for the evaluating")
# print(len(questions_set_training), "print the dimension of the training set")

df_questions_training = utils.create_df_question_based_on_pairs(df_pairs_training, df_questions)
df_questions_validation = utils.create_df_question_based_on_pairs(df_pairs_validation, df_questions)
df_questions_learning = utils.create_df_question_based_on_set(questions_set_learning, df_questions)

df_non_pairs_training = utils.make_negative_pairs(df_pairs_training, df_questions_training)
df_non_pairs_validation = utils.make_negative_pairs(df_pairs_validation, df_questions_validation)

# Create the pairs for testing
df_questions_testing = utils.create_df_question_based_on_pairs(df_pairs_testing, df_questions)
df_non_pairs_testing = utils.make_negative_pairs(df_pairs_testing, df_questions_testing)
list_testing_pairs, list_testing_labels = utils.create_dataset_process(df_pairs_testing, df_non_pairs_testing)
list_positive_testing_pairs, list_positive_testing_labels = utils.create_dataset_process(df_pairs_testing,
                                                                                         pd.DataFrame())
list_train_pairs, list_train_labels = utils.create_dataset_process(df_pairs_training, df_non_pairs_training)
list_validation_pairs, list_validation_labels = utils.create_dataset_process(df_pairs_validation,
                                                                             df_non_pairs_validation)


# Adding the hier
def use_hierarchy_mod():
    file_hier = open('../data/hierarchies/mod_hier.json')
    file_labels = open('../data/hierarchies/mod_labels.json')

    data_hier = json.load(file_hier)
    data_labels = json.load(file_labels)
    return data_hier, data_labels


def use_hierarchy_manual():
    file_hier = open('../data/hierarchies/manual_hier.json')
    file_labels = open('../data/hierarchies/manual_labels.json')

    data_hier = json.load(file_hier)
    data_labels = json.load(file_labels)
    return data_hier, data_labels


def use_hierarchy_stat():
    file_hier = open('../data/hierarchies/stat_hier.json')
    file_labels = open('../data/hierarchies/stat_labels.json')

    data_hier = json.load(file_hier)
    data_labels = json.load(file_labels)
    return data_hier, data_labels


def use_hierarchy_auto():
    file_hier = open('../data/hierarchies/automatic_hier.json')
    file_labels = open('../data/hierarchies/automatic_labels.json')

    data_hier = json.load(file_hier)
    data_labels = json.load(file_labels)
    return data_hier, data_labels


def use_embeddings_rank(rank):
    df_questions_em_body = None
    df_questions_em_title = None
    if rank == 1:
        df_questions_em_body = pd.read_pickle(
            "../data/embeddings/duplicate_questions_exp_em_body_doc2vec_full_rank1.pkl")
        df_questions_em_title = pd.read_pickle(
            "../data/embeddings/duplicate_questions_exp_em_title_doc2vec_full_rank1.pkl")
    elif rank == 2:
        df_questions_em_body = pd.read_pickle(
            "../data/embeddings/duplicate_questions_exp_em_body_doc2vec_full_rank2.pkl")
        df_questions_em_title = pd.read_pickle(
            "../data/embeddings/duplicate_questions_exp_em_title_doc2vec_full_rank2.pkl")
    elif rank == 3:
        df_questions_em_body = pd.read_pickle(
            "../data/embeddings/duplicate_questions_exp_em_body_doc2vec_full_rank3.pkl")
        df_questions_em_title = pd.read_pickle(
            "../data/embeddings/duplicate_questions_exp_em_title_doc2vec_full_rank4.pkl")
    df_questions_em_body.set_index('Id', inplace=True)
    df_questions_em_title.set_index('Id', inplace=True)
    return df_questions_em_body, df_questions_em_title


def make_data_numpy(train_data_ml, list_train_labels, validation_data_ml, list_validation_labels, learning_data_ml,
                    list_learning_labels, testing_data_ml, list_testing_labels):
    train_data_np = np.array(train_data_ml)
    train_labels_np = np.array(list_train_labels)
    validation_data_np = np.array(validation_data_ml)
    validation_labels_np = np.array(list_validation_labels)
    learning_data_np = np.array(learning_data_ml)

    learning_labels_np = np.array(list_learning_labels)
    testing_data_np = np.array(testing_data_ml)
    testing_labels_np = np.array(list_testing_labels)
    return train_data_np, train_labels_np, validation_data_np, validation_labels_np, learning_data_np, learning_labels_np, testing_data_np, testing_labels_np


def tags_scores(list_learning_pairs, list_testing_pairs, df_questions):
    ######CELL FOR CREATING THE TAGS##############
    tags_output_train_np = utils.list_tags_score_calc(list_learning_pairs, df_questions)
    tags_output_testing_np = utils.list_tags_score_calc(list_testing_pairs, df_questions)

    return (tags_output_train_np, tags_output_testing_np)


def hierachy_scores(list_learning_pairs, list_testing_pairs, df_questions):
    ######CELL FOR CREATING THE HIERARCHY##############
    data_hier_stat, data_labels_stat = use_hierarchy_stat()
    hier_stat_output_testing_np = utils.list_hier_score_calc(data_hier_stat, data_labels_stat, list_testing_pairs,
                                                             df_questions)
    hier_stat_output_train_np = utils.list_hier_score_calc(data_hier_stat, data_labels_stat, list_learning_pairs,
                                                           df_questions)
    #
    data_hier_manual, data_labels_manual = use_hierarchy_manual()
    hier_manual_output_testing_np = utils.list_hier_score_calc(data_hier_manual, data_labels_manual, list_testing_pairs,
                                                               df_questions)
    hier_manual_output_train_np = utils.list_hier_score_calc(data_hier_manual, data_labels_manual, list_learning_pairs,
                                                             df_questions)

    data_hier_mod, data_labels_mod = use_hierarchy_mod()
    hier_mod_output_testing_np = utils.list_hier_score_calc(data_hier_mod, data_labels_mod, list_testing_pairs,
                                                            df_questions)
    hier_mod_output_train_np = utils.list_hier_score_calc(data_hier_mod, data_labels_mod, list_learning_pairs,
                                                          df_questions)

    data_hier_auto, data_labels_auto = use_hierarchy_auto()
    hier_auto_output_testing_np = utils.list_hier_score_calc(data_hier_auto, data_labels_auto, list_testing_pairs,
                                                       df_questions)
    hier_auto_output_train_np = utils.list_hier_score_calc(data_hier_auto, data_labels_auto, list_learning_pairs,
                                                     df_questions)

    d_type_hier = {
        "full": (hier_auto_output_train_np, hier_auto_output_testing_np),
        "mod": (hier_mod_output_train_np, hier_mod_output_testing_np),
        "stat": (hier_stat_output_train_np, hier_stat_output_testing_np),
        "manual": (hier_manual_output_train_np, hier_manual_output_testing_np)
    }
    return d_type_hier


def evaluate_model_hier_tags(name, tags_np_tuple, training_np_tuple, testing_np_tuple, d_type_hier, type_hier):
    output_train_np, learning_labels_np = training_np_tuple
    output_testing_np, testing_labels_np = testing_np_tuple
    tags_output_train_np, tags_output_testing_np = tags_np_tuple
    hier_output_train_np, hier_output_testing_np = d_type_hier[type_hier]
    hier_train_np = np.column_stack((output_train_np, hier_output_train_np, tags_output_train_np))
    hier_testing_np = np.column_stack((output_testing_np, hier_output_testing_np, tags_output_testing_np))

    llr = LogisticRegression(random_state=42).fit(hier_train_np, learning_labels_np)
    prediction_testing = llr.predict(hier_testing_np)

    f1_score_value = f1_score(prediction_testing, testing_labels_np, average="weighted")
    accuracy = accuracy_score(prediction_testing, testing_labels_np)
    recall = recall_score(prediction_testing, testing_labels_np)
    precision = precision_score(prediction_testing, testing_labels_np)
    d_hier = {"results": "{} text + {}_hier + tags".format(name, type_hier),
              "accuracy": accuracy, "f1": f1_score_value, "recall": recall, "precision": precision,
              "coef": list(llr.coef_[0])}
    return d_hier


def evaluate_model_hier(name, train_np_tuple, testing_np_tuple, d_type_hier, type_hier):
    output_train_np, learning_labels_np = train_np_tuple
    output_testing_np, testing_labels_np = testing_np_tuple
    hier_output_train_np, hier_output_testing_np = d_type_hier[type_hier]
    hier_train_np = np.column_stack((output_train_np, hier_output_train_np))
    hier_testing_np = np.column_stack((output_testing_np, hier_output_testing_np))

    llr = LogisticRegression(random_state=42).fit(hier_train_np, learning_labels_np)
    prediction_testing = llr.predict(hier_testing_np)

    f1_score_value = f1_score(prediction_testing, testing_labels_np, average="weighted")
    accuracy = accuracy_score(prediction_testing, testing_labels_np)
    recall = recall_score(prediction_testing, testing_labels_np)
    precision = precision_score(prediction_testing, testing_labels_np)
    d_hier = {"results": "{} text + {}_hier".format(name, type_hier),
              "accuracy": accuracy, "f1": f1_score_value, "recall": recall, "precision": precision,
              "coef": list(llr.coef_[0])}
    return d_hier


def evaluate_model_tags(name, tags_np_tuple, train_np_tuple, testing_np_tuple):
    tags_output_train_np, tags_output_testing_np = tags_np_tuple
    output_train_np, learning_labels_np = train_np_tuple
    output_testing_np, testing_labels_np = testing_np_tuple
    tags_train_np = np.column_stack((output_train_np, tags_output_train_np))
    tags_testing_np = np.column_stack((output_testing_np, tags_output_testing_np))

    llr = LogisticRegression(random_state=42).fit(tags_train_np, learning_labels_np)
    # prediction_training = np.where(prediction_training >=0.5, 1, 0)
    prediction_testing = llr.predict(tags_testing_np)
    f1_score_value = f1_score(prediction_testing, testing_labels_np, average="weighted")
    accuracy = accuracy_score(prediction_testing, testing_labels_np)
    recall = recall_score(prediction_testing, testing_labels_np)
    precision = precision_score(prediction_testing, testing_labels_np)
    d_tags = {"results": "{} text + tags".format(name),
              "accuracy": accuracy, "f1": f1_score_value, "recall": recall, "precision": precision,
              "coef": list(llr.coef_[0])}
    return d_tags


def grid_search_ml_tis():

    models = {
        "GaussianNB": GaussianNB(),
        "SVM": SVC(kernel='poly', degree=3, random_state=42, probability=True),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, random_state=42),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', weights="distance"),
        "LogisticRegression": LogisticRegression(C=0.5, penalty='l1', tol=1e-8, solver='saga', random_state=42)
    }
    return models


def evaluate_model(models, d_type_hier, tags_np_tuple, learning_np_tuple, testing_np_tuple, rank):
    learning_data_np, learning_labels_np = learning_np_tuple
    testing_data_np, testing_labels_np = testing_np_tuple
    for name, model in models.items():
        start = time.time()
        model.fit(learning_data_np, learning_labels_np)
        stop = time.time()
        prediction = model.predict(testing_data_np)
        f1_score_value = f1_score(prediction, testing_labels_np, average="weighted")
        accuracy = accuracy_score(prediction, testing_labels_np)
        recall = recall_score(prediction, testing_labels_np)
        precision = precision_score(prediction, testing_labels_np)

        output_train_np = model.predict_proba(learning_data_np)
        output_testing_np = model.predict_proba(testing_data_np)
        output_testing_np = output_testing_np[:, 1]
        output_train_np = output_train_np[:, 1]

        f1_score_value = f1_score(np.where(output_testing_np >= 0.5, 1, 0), testing_labels_np, average="weighted")
        accuracy = accuracy_score(np.where(output_testing_np >= 0.5, 1, 0), testing_labels_np)
        recall = recall_score(np.where(output_testing_np >= 0.5, 1, 0), testing_labels_np)
        precision = precision_score(np.where(output_testing_np >= 0.5, 1, 0), testing_labels_np)
        d_text = {'results': "{} text".format(name),
                  'accuracy': accuracy, 'f1': f1_score_value, 'recall': recall, 'precision': precision,
                  'Training time': "{}s".format(stop - start)}
        scoring_train_tuple = (output_train_np, learning_labels_np)
        scoring_testing_tuple = (output_testing_np, testing_labels_np)
        d_tags = evaluate_model_tags(name, tags_np_tuple, scoring_train_tuple, scoring_testing_tuple)
        # d_hier_mod = evaluate_model_hier(name, output_train_np, output_testing_np, 'mod')
        # d_hier_mod_tags = evaluate_model_hier_tags(name, output_train_np, output_testing_np, 'mod')
        d_hier_mod = evaluate_model_hier(name, scoring_train_tuple, scoring_testing_tuple, d_type_hier, 'mod')
        d_hier_mod_tags = evaluate_model_hier_tags(name, tags_np_tuple, scoring_train_tuple, scoring_testing_tuple,
                                                   d_type_hier, 'mod')
        d_hier_stat = evaluate_model_hier(name, scoring_train_tuple, scoring_testing_tuple, d_type_hier, 'stat')
        d_hier_stat_tags = evaluate_model_hier_tags(name, tags_np_tuple, scoring_train_tuple, scoring_testing_tuple,
                                                    d_type_hier, 'stat')
        d_hier_manual = evaluate_model_hier(name, scoring_train_tuple, scoring_testing_tuple, d_type_hier, 'manual')
        d_hier_manual_tags = evaluate_model_hier_tags(name, tags_np_tuple, scoring_train_tuple, scoring_testing_tuple,
                                                   d_type_hier, 'manual')
        d_hier_full = evaluate_model_hier(name, scoring_train_tuple, scoring_testing_tuple, d_type_hier, 'full')
        d_hier_full_tags = evaluate_model_hier_tags(name, tags_np_tuple, scoring_train_tuple, scoring_testing_tuple,
                                                   d_type_hier, 'full')

        # d_hier_manual = evaluate_model_hier(name, output_train_np, output_testing_np, 'manual')
        # d_hier_manual_tags = evaluate_model_hier_tags(name, output_train_np, output_testing_np, 'manual')
        # d_hier_auto = evaluate_model_hier(name, output_train_np, output_testing_np, 'auto')
        # d_hier_auto_tags = evaluate_model_hier_tags(name, output_train_np, output_testing_np, 'auto')

        list_d = [d_text, d_tags, d_hier_mod, d_hier_mod_tags, d_hier_stat, d_hier_stat_tags, d_hier_manual, d_hier_manual_tags, d_hier_full, d_hier_full_tags]
        # list_d = [d_text, d_tags, d_hier_stat, d_hier_stat_tags]
        with open('../data/evaluations/{}_rank{}_evaluation.json'.format(name, rank), 'w') as fp:
            json.dump(list_d, fp)


def exp(rank, model_type):
    df_questions_em_body, df_questions_em_title = use_embeddings_rank(rank)
    train_data_ml = utils.create_dataset_ml_tis(list_train_pairs, df_questions_em_body, df_questions_em_title)
    # print("Start making the VALIDATION dataset for LR")
    validation_data_ml = utils.create_dataset_ml_tis(list_validation_pairs, df_questions_em_body,
                                                     df_questions_em_title)
    # print("Finish making the dataset for LR")
    learning_data_ml = utils.merge_2_lists(train_data_ml, validation_data_ml)
    list_learning_pairs = utils.merge_2_lists(list_train_pairs, list_validation_pairs)
    list_learning_labels = utils.merge_2_lists(list_train_labels, list_validation_labels)
    testing_data_ml = utils.create_dataset_ml_tis(list_testing_pairs, df_questions_em_body, df_questions_em_title)

    train_data_np, train_labels_np, validation_data_np, validation_labels_np, learning_data_np, learning_labels_np, testing_data_np, testing_labels_np = make_data_numpy(
        train_data_ml, list_train_labels, validation_data_ml, list_validation_labels, learning_data_ml,
        list_learning_labels, testing_data_ml, list_testing_labels)

    models = grid_search_ml_tis()
    if model_type == 'LR':
        models = {'LogisticRegression': models['LogisticRegression']}
    elif model_type == 'GNB':
        models = {'GaussianNB': models['GaussianNB']}
    elif model_type == 'DT':
        models = {'DecisionTreeClassifier': models['DecisionTreeClassifier']}
    elif model_type == 'KNN':
        models = {'KNeighborsClassifier': models['KNeighborsClassifier']}
    elif model_type == 'SVM':
        models = {'SVM': models['SVM']}
    d_type_hier = hierachy_scores(list_learning_pairs, list_testing_pairs, df_questions)
    tags_np_tuple = tags_scores(list_learning_pairs, list_testing_pairs, df_questions)
    learning_np_tuple = (learning_data_np, learning_labels_np)
    testing_np_tuple = (testing_data_np, testing_labels_np)
    evaluate_model(models, d_type_hier, tags_np_tuple, learning_np_tuple, testing_np_tuple, rank)
