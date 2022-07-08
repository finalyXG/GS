import argparse

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import tensorflow as tf

def fit_lr(X, Y, **kargs):
    logreg = LogisticRegression(**kargs).fit(X, Y)
    return logreg

def lr_eval(logreg, feat_np_tr, feat_np_val, target_tr, target_val, log=None):
    def f(feat_np, Y, logreg):
        if (sum(Y) == 0):
            return 0, len(Y), 0, 0
        y_hat = logreg.predict(feat_np)
        cnf_matrix = metrics.confusion_matrix(Y, y_hat)
        df_cnf_mx = pd.DataFrame(cnf_matrix)
        p_hat = logreg.predict_proba(feat_np)
        df_tr_pred = pd.DataFrame(p_hat, columns=['prob_0','prob_1']).assign(target=Y)

        min_pos_score = df_tr_pred.query("target==1.0")['prob_1'].min()
        false_hit_nb = df_tr_pred.query('target==0.0').shape[0]
        rdnb = df_tr_pred.query(f"prob_1 < {min_pos_score}").shape[0]
        rdr = rdnb / false_hit_nb

        return  min_pos_score, false_hit_nb, rdnb, rdr

    min_pos_score_tr, false_hit_nb_tr, rdnb_tr, rdr_tr = f(feat_np_tr, target_tr, logreg)
    min_pos_score_val, false_hit_nb_val, rdnb_val, rdr_val = f(feat_np_val, target_val, logreg)

    df_out_tr = pd.DataFrame({'min true hit score': [min_pos_score_tr],
                  'number of false hit': [false_hit_nb_tr],
                  'reduction number': [rdnb_tr],
                  'reduction rate': [rdr_tr]
                 })

    df_out_val = pd.DataFrame({'min true hit score': [min_pos_score_val],
                  'number of false hit': [false_hit_nb_val],
                  'reduction number': [rdnb_val],
                  'reduction rate': [rdr_val]
                 })

    if log==None:
        print('Training:')
        print(df_out_tr.to_markdown())
        print('Validation:')
        print(df_out_val.to_markdown())
    else:
        log.info('Training:')
        log.info('\n'+df_out_tr.to_markdown())
        log.info('Validation:')
        log.info('\n'+df_out_val.to_markdown())




def eval_node_feat_by_lr(path_tr, path_val):
    '''
    Evaluate node features by Logistic Regression.
    Example:
    path = '/export/poc/aml/L/TMAI_NWA_code/staging/stg_e/rs_l2m_2022-04-26_00.49.01.pkl'
    '''
    assert path_tr != None
    assert path_val != None
    
    df_tr = pd.read_pickle(path_tr)
    df_val = pd.read_pickle(path_val)

    feat_np_tr = np.array(df_tr['feat'].values.tolist())
    feat_np_val = np.array(df_val['feat'].values.tolist())


    # logreg = LogisticRegression().fit(feat_np_tr, df_tr['target'].values)
    logreg = fit_lr(feat_np_tr, df_tr['target'].values)

    lr_eval(logreg, feat_np_tr, feat_np_val, df_tr['target'].values, df_val['target'].values)

    
    
    
if __name__ == '__main__':
    '''
    Example:
    !python utils/check_node_feat.py \
    --path_tr /export/poc/aml/L/TMAI_NWA_code/staging/stg_e/rs_l2m_2022-04-26_00.49.01.pkl \
    --path_val /export/poc/aml/L/TMAI_NWA_code/staging/stg_e/rs_l2m_2022-04-26_10.50.51.pkl
    
    '''
    parser = argparse.ArgumentParser
    
    parser = argparse.ArgumentParser(description='Check node features.')
    parser.add_argument('--path_tr', type=str, help='path of the pickle containing training node features.')
    parser.add_argument('--path_val', type=str, help='path of the pickle containing validation node features.')

    args = parser.parse_args()
    eval_node_feat_by_lr(args.path_tr, args.path_val)



    