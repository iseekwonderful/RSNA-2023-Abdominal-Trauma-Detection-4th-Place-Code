import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from path import Path
import os
from sklearn.metrics import accuracy_score, roc_auc_score


import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics


class ParticipantVisibleError(Exception):
    pass


def normalize_probabilities_to_one(df: pd.DataFrame, group_columns: list) -> pd.DataFrame:
    # Normalize the sum of each row's probabilities to 100%.
    # 0.75, 0.75 => 0.5, 0.5
    # 0.1, 0.1 => 0.5, 0.5
    row_totals = df[group_columns].sum(axis=1)
    if row_totals.min() == 0:
        raise ParticipantVisibleError('All rows must contain at least one non-zero prediction')
    for col in group_columns:
        df[col] /= row_totals
    return df


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, verbose=True) -> float:
    '''
    Pseudocode:
    1. For every label group (liver, bowel, etc):
        - Normalize the sum of each row's probabilities to 100%.
        - Calculate the sample weighted log loss.
    2. Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    3. Calculate the sample weighted log loss for the new label group
    4. Return the average of all of the label group log losses as the final score.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # Run basic QC checks on the inputs
    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('All submission values must be numeric')

    if not np.isfinite(submission.values).all():
        raise ParticipantVisibleError('All submission values must be finite')

    if solution.min().min() < 0:
        raise ParticipantVisibleError('All labels must be at least zero')
    if submission.min().min() < 0:
        raise ParticipantVisibleError('All predictions must be at least zero')

    # Calculate the label group log losses
    binary_targets = ['bowel', 'extravasation']
    triple_level_targets = ['kidney', 'liver', 'spleen']
    all_target_categories = binary_targets + triple_level_targets

    label_group_losses = []
    sub = {}
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        solution = normalize_probabilities_to_one(solution, col_group)

        for col in col_group:
            if col not in submission.columns:
                raise ParticipantVisibleError(f'Missing submission column {col}')
        submission = normalize_probabilities_to_one(submission, col_group)
        label_group_losses.append(
            sklearn.metrics.log_loss(
                y_true=solution[col_group].values,
                y_pred=submission[col_group].values,
                sample_weight=solution[f'{category}_weight'].values
            )
        )
        if verbose:
            print(category, label_group_losses[-1])
        sub[category] = label_group_losses[-1]
    # Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    healthy_cols = [x + '_healthy' for x in all_target_categories]
    any_injury_labels = (1 - solution[healthy_cols]).max(axis=1)
    any_injury_predictions = (1 - submission[healthy_cols]).max(axis=1)
    any_injury_loss = sklearn.metrics.log_loss(
        y_true=any_injury_labels.values,
        y_pred=any_injury_predictions.values,
        sample_weight=solution['any_injury_weight'].values
    )
    sub['any'] = any_injury_loss
    if verbose:
        print('any_injury_weight', any_injury_loss)
    label_group_losses.append(any_injury_loss)
    return np.mean(label_group_losses), sub


def train_loss(outputs, labels, two_class_ce, three_class_ce, w=None):
    if not w:
        w = [1, 1, 1, 1, 1, 1]
    losses = (
            w[0] * two_class_ce(outputs[0], labels[0]) +
            w[1] * two_class_ce(outputs[1], labels[1]) +
            w[2] * three_class_ce(outputs[2], labels[2]) +
            w[3] * three_class_ce(outputs[3], labels[3]) +
            w[4] * three_class_ce(outputs[4], labels[4]) +
            w[5] * two_class_ce(outputs[5], labels[5])
    )

    return torch.mean(losses)


def pfbeta(labels, predictions, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


def macro_multilabel_auc(label, pred, gpu=-1):
    aucs = []
    for i in range(4):
        aucs.append(roc_auc_score(label[:, i], pred[:, i]))
    if gpu == 0:
        print(np.round(aucs, 4))
    return np.mean(aucs)


METRIC_FILE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

CFG = {
    'image_target_cols': [
        'pe_present_on_image',  # only image level
    ],

    'exam_target_cols': [
        'negative_exam_for_pe',  # exam level
        # 'qa_motion',
        # 'qa_contrast',
        # 'flow_artifact',
        'rv_lv_ratio_gte_1',  # exam level
        'rv_lv_ratio_lt_1',  # exam level
        'leftsided_pe',  # exam level
        'chronic_pe',  # exam level
        # 'true_filling_defect_not_pe',
        'rightsided_pe',  # exam level
        'acute_and_chronic_pe',  # exam level
        'central_pe',  # exam level
        'indeterminate'  # exam level
    ],

    'image_weight': 0.07361963,
    'exam_weights': [0.0736196319, 0.2346625767, 0.0782208589, 0.06257668712, 0.1042944785, 0.06257668712, 0.1042944785,
                     0.1877300613, 0.09202453988],
}


def rsna_torch_wloss(CFG, y_true_img, y_true_exam, y_pred_img, y_pred_exam, chunk_sizes):
    # transform into torch tensors
    y_true_img, y_true_exam, y_pred_img, y_pred_exam = torch.tensor(y_true_img, dtype=torch.float32), torch.tensor(
        y_true_exam, dtype=torch.float32), torch.tensor(y_pred_img, dtype=torch.float32), torch.tensor(y_pred_exam,
                                                                                                       dtype=torch.float32)

    # split into chunks (each chunks is for a single exam)
    y_true_img_chunks, y_true_exam_chunks, y_pred_img_chunks, y_pred_exam_chunks = torch.split(y_true_img, chunk_sizes,
                                                                                               dim=0), torch.split(
        y_true_exam, chunk_sizes, dim=0), torch.split(y_pred_img, chunk_sizes, dim=0), torch.split(y_pred_exam,
                                                                                                   chunk_sizes, dim=0)

    label_w = torch.tensor(CFG['exam_weights']).view(1, -1)
    img_w = CFG['image_weight']
    bce_func = torch.nn.BCELoss(reduction='none')

    total_loss = torch.tensor(0, dtype=torch.float32)
    total_weights = torch.tensor(0, dtype=torch.float32)

    for i, (y_true_img_, y_true_exam_, y_pred_img_, y_pred_exam_) in enumerate(
            zip(y_true_img_chunks, y_true_exam_chunks, y_pred_img_chunks, y_pred_exam_chunks)):
        exam_loss = bce_func(y_pred_exam_[0, :], y_true_exam_[0, :])
        exam_loss = torch.sum(exam_loss * label_w, 1)[
            0]  # Kaggle uses a binary log loss equation for each label and then takes the mean of the log loss over all labels.

        image_loss = bce_func(y_pred_img_, y_true_img_)
        img_num = chunk_sizes[i]
        qi = torch.sum(y_true_img_) / img_num
        image_loss = torch.sum(img_w * qi * image_loss)

        total_loss += exam_loss + image_loss
        total_weights += label_w.sum() + img_w * qi * img_num
        # print(exam_loss, image_loss, img_num);assert False

    final_loss = total_loss / total_weights
    return final_loss


def rsna_weight_loss_image_only(df, truth):
    '''
    Fill the prediction of experiment level with mean value
    For stage1 model only!

    '''
    path = METRIC_FILE_PATH / '..' / 'dataloaders/split/naive.full.stratified.5.fold.csv.zip'
    train = pd.read_csv(path)
    exam_label_mean = train[CFG['exam_target_cols']].mean(axis=0)
    df[CFG['exam_target_cols']] = exam_label_mean.values
    with torch.no_grad():
        loss = rsna_torch_wloss(CFG, truth[CFG['image_target_cols']].values, truth[CFG['exam_target_cols']].values,
                                df[CFG['image_target_cols']].values, df[CFG['exam_target_cols']].values,
                                list(truth.groupby('StudyInstanceUID', sort=False)['SOPInstanceUID'].count()))

        return loss.item()


def rsna_weight_loss_full(df, truth):
    with torch.no_grad():
        loss = rsna_torch_wloss(CFG, truth[CFG['image_target_cols']].values, truth[CFG['exam_target_cols']].values,
                                df[CFG['image_target_cols']].values, df[CFG['exam_target_cols']].values,
                                list(truth.groupby('StudyInstanceUID', sort=False)['SOPInstanceUID'].count()))

        return loss.item()


if __name__ == '__main__':
    split_df = pd.read_csv(METRIC_FILE_PATH / '..' / 'dataloaders/split/naive.full.stratified.5.fold.csv.zip')
    print(split_df.shape)
    train = split_df[split_df.fold != 0]
    valid = split_df[split_df.fold == 0]
    predicted = valid.copy()
    predicted['pe_present_on_image'] = train['pe_present_on_image'].mean()
    print(predicted.head())
    print(rsna_weight_loss_image_only(predicted, valid))
    exam_label_mean = train[CFG['exam_target_cols']].mean(axis=0)
    predicted[CFG['exam_target_cols']] = exam_label_mean.values
    print(rsna_weight_loss_full(predicted, valid))
