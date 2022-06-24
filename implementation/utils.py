import pandas as pd
import numpy as np
import random
import sklearn.utils

def merge_2_lists(l1, l2):
    l = np.concatenate([l1, l2])
    return l.tolist()


def create_set_questions_df(df_pairs):
    set_questions = set()
    for x in df_pairs['PostId']:
        if x not in set_questions:
            set_questions.add(x)
    for x in df_pairs['RelatedPostId']:
        if x not in set_questions:
            set_questions.add(x)
    return set_questions


def create_merge_data(list_pairs, list_pairs_non, list_labels_pos, list_labels_neg):
    list_data = list_pairs + list_pairs_non
    list_labels = list_labels_pos + list_labels_neg

    list_data, list_labels = sklearn.utils.shuffle(list_data, list_labels, random_state=73)
    # list_data, list_labels = sklearn.utils.shuffle(list_data, list_labels)
    return list_data, list_labels


def create_list_pairs_and_labels(df_pairs, label):
    list_pairs = []
    for i in df_pairs.index:
        list_pairs.append((df_pairs['PostId'][i], df_pairs['RelatedPostId'][i]))
    list_labels = [label for x in range(len(list_pairs))]
    return list_pairs, list_labels


def create_df_question_based_on_pairs(df_pairs, df_questions):
    set_questions = create_set_questions_df(df_pairs)
    return df_questions.loc[df_questions['Id'].isin(set_questions)]


def create_dataset_process(df_pairs_positive, df_pairs_negative):
    list_pos_pairs, list_labels_pos = create_list_pairs_and_labels(df_pairs_positive, 1)
    list_neg_pairs, list_labels_neg = create_list_pairs_and_labels(df_pairs_negative, 0)

    return create_merge_data(list_pos_pairs, list_neg_pairs, list_labels_pos, list_labels_neg)


def get_textual_info(df_questions, q_id, column):
    for i in df_questions.index:
        if df_questions['Id'][i] == q_id:
            return df_questions[column][i]


def generate_pair_embeddings(q1, q2, model_body, model_title):
    vec_body_q1 = model_body.infer_vector(get_textual_info(df_questions, q1, 'Body'))
    vec_body_q2 = model_body.infer_vector(get_textual_info(df_questions, q2, 'Body'))

    vec_title_q1 = model_title.infer_vector(get_textual_info(df_questions, q1, 'Title'))
    vec_title_q2 = model_title.infer_vector(get_textual_info(df_questions, q2, 'Title'))

    vec_q1 = np.concatenate([vec_title_q1, vec_body_q1])
    vec_q2 = np.concatenate([vec_title_q2, vec_body_q2])
    return vec_q1, vec_q2


def make_negative_pairs(df_pairs, df_questions):
    set_pairs = set()
    for i in df_pairs.index:
        x = df_pairs['PostId'][i]
        y = df_pairs['RelatedPostId'][i]
        if x < y:
            set_pairs.add((x, y))
        else:
            set_pairs.add((y, x))

    list_index_questions = df_questions.index.tolist()
    list_non_duplicate_pairs = []


    while len(list_non_duplicate_pairs) < len(df_pairs.index.tolist()):
        pair_random = random.sample(list_index_questions, 2)
        q1_id = df_questions['Id'][pair_random[0]]
        q2_id = df_questions['Id'][pair_random[1]]
        if q1_id > q2_id:
            q1_id, q2_id = q2_id, q1_id

        if (q1_id, q2_id) not in set_pairs:
            list_non_duplicate_pairs.append((q1_id, q2_id))

    df_pairs_non = pd.DataFrame(list_non_duplicate_pairs, columns=['PostId', 'RelatedPostId'])
    return df_pairs_non


def create_df_question_based_on_set(set_questions, df_questions):
    return df_questions.loc[df_questions['Id'].isin(set_questions)]


def create_positive_pair_set(df_pairs):
    set_pairs = set()
    for i in df_pairs.index:
        x = df_pairs['PostId'][i]
        y = df_pairs['RelatedPostId'][i]
        if x > y:
            x, y, = y, x
        set_pairs.add((x, y))
    return set_pairs


def filter_based_on_hier_score(hier, labels, q_1, q_2, baseline_score, df_questions):
    tags1 = df_questions.loc[df_questions['Id'] == q_1]['Tags'].to_numpy()[0]
    tags2 = df_questions.loc[df_questions['Id'] == q_2]['Tags'].to_numpy()[0]
    print("Printing the tags")
    print(tags1)
    score = compute_score(hier, labels, tags1, tags2)
    return score >= baseline_score


def filter_based_on_tags_score(q_1, q_2, baseline_score, df_questions):
    tags1 = df_questions.loc[df_questions['Id'] == q_1]['Tags'].to_numpy()[0]
    tags2 = df_questions.loc[df_questions['Id'] == q_2]['Tags'].to_numpy()[0]
    score = set(tags1).intersection(set(tags2))
    return score >= baseline_score


def filter_based_on_tags_score_v2(set_q1_tags, q_2, baseline_score, df_questions):
    # tags2 = df_questions.loc[df_questions['Id'] == q_2]['Tags'].to_numpy()[0]
    tags2 = df_questions.loc[df_questions.index == q_2]['Tags'].to_numpy()[0]
    set2 = set(tags2)
    score = len(set_q1_tags.intersection(set2))
    return score >= baseline_score


def make_recommandations(ml_model, q_id, set_pairs, baseline_score, df_questions, df_questions_learning, hier, labels,
                         df_questions_body, df_questions_title):
    # here the hierarchy can be used for filtering
    # list_pairs, list_labels = create_all_pairs_for_rec_hier(df_questions, set_pairs, q_id, baseline_score, hier, labels)
    # here the tags can be used for filtering
    # tags_q_id = df_questions.loc[df_questions['Id'] == q_id]['Tags'].to_numpy()[0]
    tags_q_id = df_questions.loc[df_questions.index == q_id]['Tags'].to_numpy()[0]
    list_pairs, list_labels = create_all_pairs_for_rec_tags(df_questions_learning, set_pairs, q_id, tags_q_id,
                                                            baseline_score)
    data_learning = create_dataset_pred_v2(list_pairs, df_questions_body, df_questions_title, q_id)
    # print(data_learning)
    predictions = ml_model.predict_proba(data_learning)[:, 1]
    # print("predictions", predictions)
    # print("I am here")
    output_data = sorted(zip(predictions, list_pairs, list_labels), reverse=True)
    # output_data.sort()
    return output_data


def labels_k_rec(ml_model, k, testing_questions, baseline_score, set_pairs, df_quesions, df_quesions_learning, hier,
                 labels, df_questions_body, df_questions_title):
    list_pred = []
    list_true = []
    list_testing_data = []
    cnt = 0
    for x in testing_questions:
        print(cnt)
        cnt = cnt + 1
        output_data = make_recommandations(ml_model, x, set_pairs, baseline_score, df_quesions, df_questions_learning,
                                           hier, labels, df_questions_body, df_questions_title)[:k]
        list_testing_data = list_testing_data + output_data
    return list_testing_data


def apply_confidence_threshold(confidence_threshold, testing_data):
    list_pred = []
    list_true = []
    for pred, p, l in testing_data:
        list_true.append(l)
        if pred >= confidence_threshold:
            list_pred.append(1)
        else:
            list_pred.append(0)
    return list_true, list_pred


def create_all_pairs_for_rec_hier(df_questions, set_positive_pairs, q_id, baseline_score, hier, labels):
    list_pairs = []
    list_labels = []
    tags_q_id = df_questions.loc[df_questions['Id'] == q_id]['Tags'].to_numpy()[0]
    for i in df_questions.index:
        q1_id = df_questions['Id'][i]
        if q_id == q1_id: continue
        q2_id = q_id
        if not filter_based_on_hier_score(hier, labels, q1_id, q2_id, baseline_score, df_questions): continue

        if q2_id < q1_id:
            q2_id, q1_id = q1_id, q2_id

        list_pairs.append((q1_id, q2_id))
        if (q1_id, q2_id) in set_positive_pairs:
            list_labels.append(1)
        else:
            list_labels.append(0)

    return list_pairs, list_labels


def create_all_pairs_for_rec_tags(df_questions, set_positive_pairs, q_id, tags_q_id, baseline_score):
    list_pairs = []
    list_labels = []
    set_tags_q_id = set(tags_q_id)
    for i in df_questions.index:
        q1_id = i
        q2_id = q_id
        if not filter_based_on_tags_score_v2(set_tags_q_id, q1_id, baseline_score, df_questions): continue

        if q2_id < q1_id:
            q2_id, q1_id = q1_id, q2_id

        list_pairs.append((q1_id, q2_id))
        # print("I am here", (q1_id, q_id))
        if (q1_id, q2_id) in set_positive_pairs:
            list_labels.append(1)
        else:
            list_labels.append(0)

    return list_pairs, list_labels


def create_embedings_dataframe(df_questions, model, column):
    list_ids = []
    list_em = []
    for i in df_questions.index:
        x = df_questions['Id'][i]
        text = df_questions[column][i]
        vec = model.infer_vector(text)
        list_em.append(vec)
        list_ids.append(x)

    new_column = column + " Em"
    data = {'Id': list_ids, new_column: list_em}
    df = pd.DataFrame(data)
    return df


def create_post_embeddings_dataframe(df_questions, model_body, model_title):
    list_ids = []
    list_em = []
    for i in df_questions.index:
        x = df_questions['Id'][i]
        title_text = df_questions['Title'][i]
        body_text = df_questions['Body'][i]
        title = model_title.infer_vector(title_text)
        body = model_body.infer_vector(body_text)
        list_em.append(np.concatenate((title, body), axis=None))
        list_ids.append(x)
        # print(list_em)
        # break
    new_column = 'Embeddings'
    data = {'Id': list_ids, new_column: list_em}
    df = pd.DataFrame(data)
    return df


def create_dataset_ml_tis(list_pairs, df_questions_body, df_questions_title):
    list_input = []
    for x, y in list_pairs:
        if x > y: x, y = y, x

        t1 = df_questions_title[df_questions_title.index == x]['Title Em'].to_numpy()[0]
        t2 = df_questions_title[df_questions_title.index == y]['Title Em'].to_numpy()[0]
        b1 = df_questions_body[df_questions_body.index == x]['Body Em'].to_numpy()[0]
        b2 = df_questions_body[df_questions_body.index == y]['Body Em'].to_numpy()[0]

        l = np.concatenate((t1, b1, t2, b2), axis=None)

        list_input.append(l)

    return list_input


def create_dataset_pred_v2(list_pairs, df_questions_body, df_questions_title, q_id):
    t = df_questions_title[df_questions_title.index == q_id]['Title Em'].to_numpy()[0]
    b = df_questions_body[df_questions_body.index == q_id]['Body Em'].to_numpy()[0]
    list_input = []
    for x, y in list_pairs:
        t1 = None
        b1 = None
        b2 = None
        t2 = None
        if x == q_id:
            t1 = t
            t2 = df_questions_title[df_questions_title.index == y]['Title Em'].to_numpy()[0]
            b1 = b
            b2 = df_questions_body[df_questions_body.index == y]['Body Em'].to_numpy()[0]
        else:
            t2 = t
            t1 = df_questions_title[df_questions_title.index == y]['Title Em'].to_numpy()[0]
            b2 = b
            b1 = df_questions_body[df_questions_body.index == y]['Body Em'].to_numpy()[0]
        if x > y:
            x, y = y, x
            t1, t2 = t2, t1
            b1, b2 = b2, b1

        l = np.concatenate((t1, b1, t2, b2), axis=None)
        list_input.append(l)

    return list_input


def sim_score_em_pair(q1, q2, df_questions_em):
    vec1 = df_questions_em[df_questions_em.index == q1]['Embeddings'].to_numpy()[0]
    vec2 = df_questions_em[df_questions_em.index == q2]['Embeddings'].to_numpy()[0]
    # print(vec1)
    cos_sim = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return cos_sim


def create_output_data_sim(list_pairs, list_labels, df_questions_em):
    list_scores = []

    for x, y in list_pairs:
        list_scores.append(sim_score_em_pair(x, y, df_questions_em))
    # output_data = zip(list_scores, list_pairs, list_labels)
    output_data = sorted(zip(list_scores, list_pairs, list_labels), reverse=True)
    return output_data






################Computer Hierarchy scores
def dfs_start(dict_hier, dict_labels, x, y):
    score = 0
    if x in dict_hier['value']['content'] and y in dict_hier['value']['content']:
        score = float(dict_labels[str(dict_hier['value']['uniqueId'])]['weight'])
    for child in dict_hier['children']:
        score_child = dfs_start(child, dict_labels, x, y)
        score = max(score_child, score)
    return score


def compute_score(dict_hier, dict_labels, tags1, tags2):
    score = 0
    for x in tags1:
        for y in tags2:
            score = max(score, dfs_start(dict_hier, dict_labels, x, y))
    return score


def hier_score_calc(hier, labels, q_1, q_2, df_questions):
    tags1 = df_questions.loc[df_questions['Id'] == q_1]['Tags'].to_numpy()[0]
    tags2 = df_questions.loc[df_questions['Id'] == q_2]['Tags'].to_numpy()[0]
    return compute_score(hier, labels, tags1, tags2)

def tags_score_calc(q_1, q_2, df_questions):
    tags1 = df_questions.loc[df_questions['Id'] == q_1]['Tags'].to_numpy()[0]
    tags2 = df_questions.loc[df_questions['Id'] == q_2]['Tags'].to_numpy()[0]
    return len(set(tags1).intersection(set(tags2))) / len(set(tags1).union(set(tags2)))

def list_hier_score_calc(hier, labels, list_pairs, df_questions):
    list_scores = []
    for x, y in list_pairs:
        list_scores.append(hier_score_calc(hier, labels, x, y, df_questions))
    return np.array(list_scores)


def list_tags_score_calc(list_pairs, df_questions):
    list_scores = []
    for x, y in list_pairs:
        list_scores.append(tags_score_calc(x, y, df_questions))
    return np.array(list_scores)
