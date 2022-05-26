import argparse
from datetime import datetime
import json
import logging
import random
import sys

import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from networkx.readwrite import json_graph
import tqdm
import plotnine as p9, plotnine.data
from neo4j import GraphDatabase
log = logging.getLogger('nwa.final.analysis.com')
PROJ_PATH = '/mnt/c/Users/Laurence_Liu/Documents/Regtics_proj/NWA_AI'
neo4j_log = logging.getLogger("neo4j")
neo4j_log.setLevel(logging.CRITICAL)


def run_neo4j_query(query, driver):
    rs = None
    with driver.session() as session:
        _rs = session.run(
        query)
        rs = _rs.data()
    return rs

def lassert(args):
    assert args
    return ''


exp_config = {
    'global': {
        'yyyymmdd_HHMM': datetime.today().strftime('%Y%m%d_%H%M'),
        'PROJ_PATH' : PROJ_PATH,
        'set_G_by_json': lambda : (
            G_data := json.load(open(args.graph_sage_G_path)),
            G := json_graph.node_link_graph(G_data),
            exp_config['global'].update({'G':G})
        ),
        'set_logging_info': lambda : (
            proj_path := exp_config['global']['PROJ_PATH'],
            yyyymmdd_HHMM := exp_config['global']['yyyymmdd_HHMM'],
            staging := f"{proj_path}/staging/tmp_t/",
            Path(f"{staging}{yyyymmdd_HHMM}").mkdir(parents=True, exist_ok=True),
            logging_path := f'{staging}{yyyymmdd_HHMM}/neo4j_explore_output.log',
            # logging.basicConfig(filename=logging_path, filemode='w', level=logging.DEBUG),
            exp_config['global'].update({'staging_path': f'{staging}{yyyymmdd_HHMM}'}),
            hdlr := logging.StreamHandler(),
            fhdlr := logging.FileHandler(f"{logging_path}", mode='w', encoding='utf-8'),
            formatter := logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            hdlr.setFormatter(formatter),
            fhdlr.setFormatter(formatter),
            log.addHandler(hdlr),
            log.addHandler(fhdlr),
            log.setLevel(logging.DEBUG),
        ),
        'run_step_ls':[
            'set_G_by_json',
            'set_logging_info',
        ],
        'run_exp_ls' : [
            'check_neo4j_data_by_G',
            'View_df_degree', 
            'view_tr_te_0_1_count',
            'view_dg1_dg2_dist',
            'view_dg1_dg1(pos)_dist',
            'gen_image_about_degree^1_degree^1(pos)',
            'gen_hist_about_true_false_hit',
            'gen_image_about_degree^1_degree^2',
        ],
    },
    'check_neo4j_data_by_G':{
        'help': 'Cross check neo4j data via json data (G).',
        'type': "cypher query",
        'cypher_query_ls': [
            'match (n) return count(n)',
            'match ()-[r]->() return count(r)',
        ],
        'cypher_return_callback': lambda _rs_total_nodes, _rs_total_edges : (
            G := exp_config['global']['G'],
            log.info(f"Total number of nodes: {_rs_total_nodes[0]['count(n)']}"),
            log.info(f"Total number of edges: {_rs_total_edges[0]['count(r)']}"),
            lassert(_rs_total_nodes[0]['count(n)'] == len(G.nodes)),
            lassert(_rs_total_edges[0]['count(r)'] == len(G.edges)),
        ),
    },
    'View_df_degree':{
        'help': 'Cross check neo4j data via json data (G).',
        'example output': '''
            Expected:
            |      |   nwa_id |   target |   degree^1 | type   |   degree^2 |
            |-----:|---------:|---------:|-----------:|:-------|-----------:|
            |    0 |        0 |        0 |          8 | train  |        207 |
            |    1 |        1 |        0 |         13 | train  |        387 |
        ''',
        'type': "cypher query",
        'cypher_query_ls': [
            '''
            match (n{test:false, val:false})-[*1..1]-(t{test:false, val:false})
            return n.nwa_id, n.target, count(t.nwa_id)
            ''',
            '''
            match (n{test:false, val:true})-[*1..1]-(t{test:false, val:true})
            return n.nwa_id, n.target, count(t.nwa_id)
            ''',
            '''
            match (n{test:true, val:false})-[*1..1]-(t{test:true, val:false})
            return n.nwa_id, n.target, count(t.nwa_id)
            ''',
            # Found 2nd order degree
            '''
            match (n{test:false, val:false})-[*2..2]-(t{test:false, val:false})
            return n.nwa_id, count(t.nwa_id)
            ''',
            '''
            match (n{test:false, val:true})-[*2..2]-(t{test:false, val:true})
            return n.nwa_id, count(t.nwa_id)
            ''',
            '''
            match (n{test:true, val:false})-[*2..2]-(t{test:true, val:false})
            return n.nwa_id, count(t.nwa_id)
            ''',
        ],
        'cypher_return_callback': lambda q_pow1_tr, q_pow1_val, q_pow1_te, q_pow2_tr, q_pow2_val, q_pow2_te : (
            G := exp_config['global']['G'],
            df_pow1_tr := pd.DataFrame(q_pow1_tr)\
                .rename(columns={'n.nwa_id':'nwa_id', 'count(t.nwa_id)':'degree^1', 'n.target':'target'}),
            df_pow1_val := pd.DataFrame(q_pow1_val)\
                .rename(columns={'n.nwa_id':'nwa_id', 'count(t.nwa_id)':'degree^1', 'n.target':'target'}),
            df_pow1_te := pd.DataFrame(q_pow1_te)\
                .rename(columns={'n.nwa_id':'nwa_id', 'count(t.nwa_id)':'degree^1', 'n.target':'target'}),
            df_pow2_tr := pd.DataFrame(q_pow2_tr)\
                .rename(columns={'n.nwa_id':'nwa_id', 'count(t.nwa_id)':'degree^2'}),
            df_pow2_val := pd.DataFrame(q_pow2_val)\
                .rename(columns={'n.nwa_id':'nwa_id', 'count(t.nwa_id)':'degree^2'}),
            df_pow2_te := pd.DataFrame(q_pow2_te)\
                .rename(columns={'n.nwa_id':'nwa_id', 'count(t.nwa_id)':'degree^2'}),
            df_dg1 := pd.concat([df_pow1_tr.assign(type='train'),
                df_pow1_val.assign(type='val'),
                df_pow1_te.assign(type='test')]),
            df_dg2 := pd.concat([df_pow2_tr.assign(type='train'),
                df_pow2_val.assign(type='val'),
                df_pow2_te.assign(type='test')]),
            df_dg := pd.merge(left=df_dg1, right=df_dg2.drop(columns=['type']), on='nwa_id', how='left')\
                .fillna(value={'degree^2':0}),
            exp_config['global'].update({'df_dg':df_dg}),
        ),
    },
    'view_tr_te_0_1_count':{
        'help': 'viwe neo4j data [train/test/val] x [0/1] counts.',
        'example output': '''
        Excepted:
        All nodes (i.e., degree>=0) target distribution
        | type   |   0.0 |   1.0 |
        |:-------|------:|------:|
        | test   |   257 |   655 |
        | train  |  4424 |  1108 |
        | val    |    44 |   571 |
        ''',
        'type': "cypher query",
        'cypher_query_ls': [
            'match (n{test:false, val:false, target:0.0}) return count(n) as count_tr_0',
            'match (n{test:false, val:false, target:1.0}) return count(n) as count_tr_1',
            'match (n{test:false, val:true, target:0.0}) return count(n) as count_val_0',
            'match (n{test:false, val:true, target:1.0}) return count(n) as count_val_1',
            'match (n{test:true, val:false, target:0.0}) return count(n) as count_te_0',
            'match (n{test:true, val:false, target:1.0}) return count(n) as count_te_1',
        ],
        'cypher_return_callback': lambda q_tr_0, q_tr_1, q_val_0, q_val_1, q_te_0, q_te_1: (
            df_count_tr_0 := pd.DataFrame(q_tr_0),
            df_count_tr_1 := pd.DataFrame(q_tr_1),
            df_count_val_0 := pd.DataFrame(q_val_0),
            df_count_val_1 := pd.DataFrame(q_val_1),
            df_count_te_0 := pd.DataFrame(q_te_0),
            df_count_te_1 := pd.DataFrame(q_te_1),
            df_pivot_count_all_nodes := pd.DataFrame({
                'type':['test','train','val'], 
                '0.0':[df_count_te_0.values[0][0], df_count_tr_0.values[0][0], df_count_val_0.values[0][0]], 
                '1.0':[df_count_te_1.values[0][0], df_count_tr_1.values[0][0], df_count_val_1.values[0][0]] }).set_index('type'),
            log.info('\n'+df_pivot_count_all_nodes.to_markdown()),
        )
    }
}



exp_config['view_dg1_dg2_dist'] = {
    'help':'View degree^1 and degree^2 distribution by groups of ["type", "target"]',
    'input_looks_like': "",
    'type': 'run lambda',
    'lambda': lambda :(
        df_dg := exp_config['global']['df_dg'],
        df := pd.pivot_table(df_dg, values=['degree^1','degree^2'], index=['type', 'target'],
            aggfunc={
                'degree^1':['count', np.mean, np.std],
                'degree^2':['count', np.mean, np.std],
        }),
        log.info('\n'+df.T.to_markdown()),
    )
}


exp_config['view_dg1_dg1(pos)_dist'] = {
    'help':'View degree^1 and degree^1(pos) distribution by groups of ["type", "target"]',
    'input_looks_like': '''
    |    |   nwa_id |   target |   degree^1 |   degree^1(pos) | type   |
    |---:|---------:|---------:|-----------:|----------------:|:-------|
    |  0 |        0 |        0 |          8 |               0 | train  |
    |  1 |        1 |        0 |         13 |               5 | train  |
    ''',
    'type': 'cypher query',
    "cypher_query_ls":[
        '''
        match (n{test:false, val:false})--(t{test:false, val:false})
        return n.nwa_id, n.target, count(t), sum(t.target)
        ''',
        '''
        match (n{test:false, val:true})--(t{test:false, val:true})
        return n.nwa_id, n.target, count(t), sum(t.target)
        ''',
        '''
        match (n{test:true, val:false})--(t{test:true, val:false})
        return n.nwa_id, n.target, count(t), sum(t.target)
        ''',
    ],
    'cypher_return_callback': lambda q_tfr_tr, q_tfr_val, q_tfr_te: (
        df_tfr_tr := pd.DataFrame(q_tfr_tr)\
            .rename(columns={'n.nwa_id':'nwa_id', 'n.target':'target', 'count(t)':'degree^1', 'sum(t.target)':'degree^1(pos)'}),
        df_tfr_val := pd.DataFrame(q_tfr_val)\
            .rename(columns={'n.nwa_id':'nwa_id', 'n.target':'target', 'count(t)':'degree^1', 'sum(t.target)':'degree^1(pos)'}),
        df_tfr_te := pd.DataFrame(q_tfr_te)\
            .rename(columns={'n.nwa_id':'nwa_id', 'n.target':'target', 'count(t)':'degree^1', 'sum(t.target)':'degree^1(pos)'}),
        df_tfr := pd.concat([
            df_tfr_tr.assign(type='train'),
            df_tfr_val.assign(type='val'),
            df_tfr_tr.assign(type='test')
        ]),
        df_pv := pd.pivot_table(
            df_tfr.assign(**{'degree^1(pos/all)': lambda x: x['degree^1(pos)'] / x['degree^1']  }), 
            index=['type','target'], values=['degree^1', 'degree^1(pos)', 'degree^1(pos/all)'],
            aggfunc={
                'degree^1':['count',np.mean, np.std],
                'degree^1(pos)':['count',np.mean, np.std],
                'degree^1(pos/all)':['count',np.mean, np.std],
            }
        ),
        exp_config['global'].update({'df_view_dg1_dg1(pos)_dist': df_tfr}),
        log.info('\n'+df_pv.T.to_markdown())
    )
}


exp_config['gen_image_about_degree^1_degree^1(pos)'] = {
    'help':'Generate image about degree^1 and degree^1(pos) ["type", "target"]',
    'input_looks_like': "",
    'type': 'run lambda',
    'lambda': lambda :(
        df_tfr := exp_config['global']['df_view_dg1_dg1(pos)_dist'],
        tmp_plot := p9.ggplot(df_tfr.assign(target=lambda x:x['target'].astype(str)) ) +\
            p9.aes(x="degree^1", y="degree^1(pos)",fill="target") +\
            p9.geom_point() +\
            p9.geom_abline(intercept = 0, slope = 1, size = 0.5, color='blue',linetype='dashed') +\
            p9.facet_wrap('~target + type') +\
            p9.theme(figure_size=(7, 4)),
        tmp_plot.save(f"{exp_config['global']['staging_path']}/degree1vs1(pos).svg", height=6, width=10)
    )
}


exp_config['gen_hist_about_true_false_hit'] = {
    'help':'Generate histogram about true and false positive distribution.',
    'input_looks_like': '''
    |    |   nwa_id |   target |   degree^1 | type   |   degree^2 |
    |---:|---------:|---------:|-----------:|:-------|-----------:|
    |  0 |        0 |        0 |          8 | train  |        207 |
    |  1 |        1 |        0 |         13 | train  |        387 |
    ''',
    'type': 'run lambda',
    'lambda': lambda :(
        tmp_plot := p9.ggplot(exp_config['global']['df_dg']) +\
            p9.aes(x="degree^1", fill="type") +\
            p9.geom_histogram(bins=12, color="#333333", alpha=0.6, position = 'identity') +\
            p9.facet_wrap('~target+type') +\
            p9.theme(figure_size=(8, 5)),
        tmp_plot.save(f"{exp_config['global']['staging_path']}/nodes_ori.svg", height=6, width=10)
        ),
}
        

exp_config['gen_image_about_degree^1_degree^2'] = {
    'help':'Generate histogram about true and false positive distribution.',
    'input_looks_like': '''
    |    |   nwa_id |   target |   degree^1 | type   |   degree^2 |
    |---:|---------:|---------:|-----------:|:-------|-----------:|
    |  0 |        0 |        0 |          8 | train  |        207 |
    |  1 |        1 |        0 |         13 | train  |        387 |
    ''',
    'type': 'run lambda',
    'lambda': lambda :(

        tmp_plot := p9.ggplot(exp_config['global']['df_dg']) +\
            p9.aes(x="degree^1", y="degree^2",fill="type") +\
            p9.geom_point() +\
            p9.facet_wrap('~type') +\
            p9.theme(figure_size=(6, 2)),
        tmp_plot.save(f"{exp_config['global']['staging_path']}/degree1vs2.svg", height=4, width=10)
        ),
}




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create Neo4j analysis report.')
    parser.add_argument('--graph_sage_G_path', type=str, help='Path of json file of GraphSAGE G.',
        default='/mnt/c/Users/Laurence_Liu/Documents/Regtics_proj/NWA_AI/example_data/livi-test-data-20220511-G.json')
    parser.add_argument('--neo4j_uri', type=str, help='Uri of neo4j DB', default='bolt://localhost:7687')
    parser.add_argument('--neo4j_auth_u', type=str, help='User of neo4j DB', default='neo4j')
    parser.add_argument('--neo4j_auth_pw', type=str, help='Password of neo4j DB', default='n1234567n')
    parser.add_argument('--save_to_folder_by_date', type=str, help='Date value yyyymmdd_HHMM as saved folder name', default=None)
    args = parser.parse_args()

    if args.save_to_folder_by_date:
        exp_config['global']['yyyymmdd_HHMM'] = args.save_to_folder_by_date

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_auth_u, args.neo4j_auth_pw), encrypted=False)



    for e in exp_config['global']['run_step_ls']:
        exp_config['global'][e]()
            
    for e in exp_config['global']['run_exp_ls']:
        exp = exp_config[e]
        kind = exp.setdefault('type', '')
        if kind == "cypher query":
                q_rs = [run_neo4j_query(q, driver) for q in exp['cypher_query_ls']]
                cb_rs = exp['cypher_return_callback'](*q_rs)
        if kind == 'run lambda':
                exp['lambda']()
