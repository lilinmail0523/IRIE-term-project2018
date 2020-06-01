import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import ast
from gensim import corpora
import os
from  bm25_util import *
from tqdm import tqdm
import pandas as pd
from scipy.stats import rankdata
from gensim.models import TfidfModel




cur_dir = os.getcwd()

#read excel list(string) to python list. nan will get error.
def test_nan(subject):
    try:
        test_nan = ast.literal_eval(subject)
    except :
        test_nan = []
    return test_nan

#build a Dictionary for query, by thier tf-idf weight, higher value will get higher frequency in query
def get_query_weight(query):
    #build Dictionary
    dct = corpora.Dictionary(query)
    #represent word by ID
    corpus = [dct.doc2bow(line) for line in query]
    #tf-idf model
    model = TfidfModel(corpus)

    return model,corpus, dct

#by tf-idf value , highest term is given three more times to highlight its importance, and so do second, third term
def get_weighted_query(model, corpus,ques_num, query, dct):
    #get tf-idf value
    vector = model[corpus[ques_num]]
    #sort
    vector.sort(key=lambda tup: tup[1], reverse=True)
    #highest is given three term, second and third are given two and one more times
    puls_num = 3
    for item , value in vector[:3]:
        #given word
        query = query + [dct.get(item)] * puls_num
        puls_num = puls_num - 1
    return query

# bm25 model
def get_bm25_model(ans):


    #get model
    bm25_model = BM25(ans)
    #get avg idf
    average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())

    return bm25_model, average_idf
# get bm25 scpre
def get_bm25_score(bm25_model, query, average_idf, ques_num):
    #get scores
    scores = bm25_model.get_scores(query, average_idf)
    #only use  QUESTION range of scores
    temp_score = scores[ques_num:ques_num+ 10]
    #rank scores
    rank = (len(temp_score) - rankdata(temp_score).astype(int)).tolist()
    #print(rank)
    return rank, scores[ques_num:ques_num+ 10]


def subtaskA(data, outputname):
    #fetch data
    THREAD_SEQUENCE = data["THREAD_SEQUENCE"].tolist()
    RelQBody = data["RelQBody"].tolist()
    RelQSubject = data["RelQSubject"].tolist()
    RELC_ID = data["RELC_ID"].tolist()
    RelCText = data["RelCText"].tolist()
    #scores
    score_record = []
    #rank
    rank = []
    #answer set
    ans = []
    #question set
    que = []
    #RELEVANCE
    RELC_RELEVANCE2RELQ = []
    for ques_num in range(0,  len(THREAD_SEQUENCE)):
        #collect all question (body and subject)and answer
        ans.append(test_nan(RelCText[ques_num]))
        que.append(test_nan(RelQBody[ques_num ]) + test_nan(RelQSubject[ques_num ]))
        #RELEVANCE  is useless for evaluating MAP
        RELC_RELEVANCE2RELQ.append('true')
    #build query , ans model
    model, avgidf = get_bm25_model(ans)
    query_model, query_corpus, query_dct = get_query_weight(que)
    for ques_num in tqdm(range(0,  len(THREAD_SEQUENCE), 10), ascii = True):
        #get weighted query by tf-idf weighted
        weighted_query = get_weighted_query(query_model, query_corpus,ques_num, test_nan(RelQBody[ques_num ]) + test_nan(RelQSubject[ques_num ]), query_dct)

        #get score, rank
        rank_temp , score_temp = get_bm25_score( model, weighted_query, avgidf, ques_num)
        score_record = score_record + score_temp
        rank = rank + rank_temp

    #save output
    subtaskA = pd.DataFrame({'THREAD_SEQUENCE' : THREAD_SEQUENCE, 'RELC_ID' : RELC_ID, 'RELC_RELEVANCE2RELQ' : RELC_RELEVANCE2RELQ, 'bm25' : score_record, 'rank' : rank})
    subtaskA = subtaskA[['THREAD_SEQUENCE', 'RELC_ID', 'rank','bm25', 'RELC_RELEVANCE2RELQ']]

    subtaskA.to_csv(os.path.join(cur_dir, outputname), header=None, index=None, sep=' ', mode='a')


def subtaskB(data, outputname):
    #fetch data

    data = data.drop_duplicates(subset = ["THREAD_SEQUENCE"])
    ORGQ_ID = data["ORGQ_ID"].tolist()
    OrgQSubject = data["OrgQSubject"].tolist()
    OrgQBody =  data["OrgQBody"].tolist()
    THREAD_SEQUENCE = data["THREAD_SEQUENCE"].tolist()
    RelQBody = data["RelQBody"].tolist()
    RelQSubject = data["RelQSubject"].tolist()
    #RELEVANCE
    RELQ_RELEVANCE2ORGQ = []
    #scores
    score_record = []
    #question set
    qus = []
    #origin question set
    org = []
    #rank

    rank = []
    for ques_num in range(0,  len(ORGQ_ID)):
        #collect all question(body and subject) and origin(body and subject)
        qus.append(test_nan(RelQSubject[ques_num ])+test_nan(RelQBody[ques_num ]))
        org.append(test_nan(OrgQBody[ques_num ]) + test_nan(OrgQSubject[ques_num ]))
        #RELEVANCE  is useless for evaluating MAP

        RELQ_RELEVANCE2ORGQ.append('true')
    #build query , question model
    model, avgidf = get_bm25_model(qus)
    query_model, query_corpus, query_dct = get_query_weight(org)

    for ques_num in tqdm(range(0,  len(ORGQ_ID), 10), ascii = True):

        #get weighted query by tf-idf weighted

        weighted_query = get_weighted_query(query_model, query_corpus,ques_num,test_nan(OrgQBody[ques_num ]) + test_nan(OrgQSubject[ques_num ]), query_dct)

        #get score, rank

        rank_temp , score_temp = get_bm25_score( model, weighted_query, avgidf, ques_num)
        score_record = score_record + score_temp
        rank = rank + rank_temp

    #save output

    subtaskB = pd.DataFrame({'THREAD_SEQUENCE' : THREAD_SEQUENCE, 'bm25' : score_record, 'ORGQ_ID': ORGQ_ID, 'RELQ_RELEVANCE2ORGQ' : RELQ_RELEVANCE2ORGQ, 'rank' : rank})
    subtaskB = subtaskB[['ORGQ_ID', 'THREAD_SEQUENCE', 'rank', 'bm25', 'RELQ_RELEVANCE2ORGQ']]

    subtaskB.to_csv(os.path.join(cur_dir, outputname), header=None, index=None, sep=' ', mode='a')



def subtaskC(data, outputname):
    #fetch data


    ORGQ_ID = data["ORGQ_ID"].tolist()
    OrgQSubject = data["OrgQSubject"].tolist()
    OrgQBody =  data["OrgQBody"].tolist()
    RELC_ID = data["RELC_ID"].tolist()
    #RELEVANCE

    RELC_RELEVANCE2RELQ = []
    RelCText = data["RelCText"].tolist()
    #scores
    score_record = []
    #answer set

    ans = []
    #origin question set

    org = []

    rank = []
    for ques_num in range(0,  len(ORGQ_ID)):
        #collect all question(body and subject) and origin(body and subject)

        ans.append(test_nan(RelCText[ques_num]))
        org.append(test_nan(OrgQBody[ques_num ]) + test_nan(OrgQSubject[ques_num ]))
        #RELEVANCE  is useless for evaluating MAP

        RELC_RELEVANCE2RELQ.append('true')
    #build query , question model
    query_model, query_corpus, query_dct = get_query_weight(org)
    model, avgidf = get_bm25_model(ans)
    for ques_num in tqdm(range(0,  len(ORGQ_ID), 10), ascii = True):
        #get weighted query by tf-idf weighted

        weighted_query = get_weighted_query(query_model, query_corpus,ques_num, test_nan(OrgQBody[ques_num ]) + test_nan(OrgQSubject[ques_num ]), query_dct)

        #get score, rank
        rank_temp , score_temp = get_bm25_score( model, weighted_query, avgidf, ques_num)
        score_record = score_record + score_temp
        rank = rank + rank_temp

    #save output
    subtaskC = pd.DataFrame({'ORGQ_ID' : ORGQ_ID, 'RELC_ID' : RELC_ID, 'RELC_RELEVANCE2RELQ' : RELC_RELEVANCE2RELQ, 'bm25' : score_record, 'rank' : rank})
    subtaskC = subtaskC[['ORGQ_ID', 'RELC_ID', 'rank', 'bm25', 'RELC_RELEVANCE2RELQ']]

    subtaskC.to_csv(os.path.join(cur_dir, outputname), header=None, index=None, sep=' ', mode='a')



def excu_bm25():
    #data fetch
    data = pd.read_csv(os.path.join(cur_dir, "test_use.csv"), engine='python')
    print("---bm25 subtaskA----")
    subtaskA(data, "bm25_subtaskA.pred")
    print("---bm25 subtaskB----")
    subtaskB(data, "bm25_subtaskB.pred")
    print("---bm25 subtaskC----")

    subtaskC(data, "bm25_subtaskC.pred")
