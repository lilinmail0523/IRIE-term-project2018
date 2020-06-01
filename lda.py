import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora
import os
from tqdm import tqdm
import pandas as pd
from scipy.stats import rankdata
from gensim.models import TfidfModel


from gensim.matutils import sparse2full
from gensim.models import LdaModel

from scipy import spatial
import ast
import numpy as np

cur_dir = os.getcwd()
num_topics = 50
tf_idf_weight = True

#read excel list(string) to python list. nan will get error.
def test_nan(subject):
    try:
        test_nan = ast.literal_eval(subject)
    except :
        test_nan = []
    return test_nan


#Hellinger similarity
def l2_distance(query, ans):
    scores = []
    for i in ans:

        scores.append(1.0 - spatial.distance.euclidean(query, i))
    rank = (len(scores) - rankdata(scores).astype(int)).tolist()

    return scores, rank




def Hellinger(query, ans):
    scores = []
    for i in ans:

        scores.append( np.sqrt(np.sum((np.sqrt(query) - np.sqrt(i)) ** 2)) /np.sqrt(2))
    rank = (len(scores) - rankdata(scores).astype(int)).tolist()

    return scores, rank

#cos_sim similarity
def cos_sim(query, ans):
    scores = []
    for i in ans:
        if np.sum(i) == 0:
            scores.append(0.0)
        else:
            scores.append( 1.0 - spatial.distance.cosine(query, i))
    rank = (len(scores) - rankdata(scores).astype(int)).tolist()

    return scores, rank


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


def get_lda_model(ans):

    dic = corpora.Dictionary(ans)
    #print((dic))


    doc_vectors = [dic.doc2bow(text) for text in ans]

    ans_tfidf_model = TfidfModel(doc_vectors)
    #https://stackoverflow.com/questions/45310925/how-to-get-a-complete-topic-distribution-for-a-document-using-gensim-lda
    lda = LdaModel(corpus=doc_vectors, id2word=dic, num_topics= num_topics,  minimum_probability=0.0, random_state=0)

    #print(len(doc_vectors))
    #https://stackoverflow.com/questions/45310925/how-to-get-a-complete-topic-distribution-for-a-document-using-gensim-lda



    return lda, dic, doc_vectors, ans_tfidf_model


def get_similarity_score(lda, dic ,doc,query,  ques_num, ans_tfidf):
    query_weight = np.zeros(num_topics)
    document_weight = np.zeros((10, num_topics))
    query_bow = dic.doc2bow(query)
    if(tf_idf_weight):
        ques_vector = ans_tfidf[query_bow]
        for i, j in ques_vector:
            #print("============================================================")
            #print(lda.get_term_topics(i,minimum_probability=0.0 ))
            for topic_id, topic_weight in lda.get_term_topics(i,minimum_probability=0.0 ):
                query_weight[topic_id] = query_weight[topic_id] + topic_weight * j
        for doc_num in range(ques_num, ques_num+10):
            doc_vector = ans_tfidf[ doc[doc_num]]
            for i, j in doc_vector:
                for topic_id, topic_weight in lda.get_term_topics(i,minimum_probability=0.0 ):
                    document_weight[doc_num - ques_num][topic_id] = document_weight[doc_num - ques_num][topic_id] + topic_weight * j

    else:
        #print(query_bow)
        for i, j in query_bow:
            #print("============================================================")
            #print(lda.get_term_topics(i,minimum_probability=0.0 ))
            for topic_id, topic_weight in lda.get_term_topics(i,minimum_probability=0.0 ):
                query_weight[topic_id] = query_weight[topic_id] + topic_weight * j

            #print("============================================================")
        #print("============================================================")
        #print(query_weight)
        #print("============================================================")

        #print(doc[ques_num])
        for doc_num in range(ques_num, ques_num+10):
            for i, j in doc[doc_num]:
                for topic_id, topic_weight in lda.get_term_topics(i,minimum_probability=0.0 ):
                    document_weight[doc_num - ques_num][topic_id] = document_weight[doc_num - ques_num][topic_id] + topic_weight * j


    """
    que_topic = lda.get_document_topics(query_bow, minimum_probability=0.0)
    que_topic = [item[1] for item in que_topic]

    doc_topic = []
    for i in doc[ques_num:ques_num+10]:
        temp = lda.get_document_topics(i, minimum_probability=0.0)
        doc_topic.append( [item[1] for item in temp])
    """
    #print(dic_temp)
    #query_dic = dic.doc2bow(query)
    #query_lda = [item[1] for item in lda[query_dic]]
    #print(doc_topic)
    return  cos_sim(query_weight,document_weight)



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
    model , dic, doc , ans_tfidf= get_lda_model(ans)
    query_model, query_corpus, query_dct = get_query_weight(que)

    for ques_num in tqdm(range(0,  len(THREAD_SEQUENCE), 10), ascii = True):
        #get weighted query by tf-idf weighted
        weighted_query = get_weighted_query(query_model, query_corpus,ques_num, test_nan(RelQBody[ques_num ]) + test_nan(RelQSubject[ques_num ]), query_dct)

        #get score, rank
        temp_score , temp_rank = get_similarity_score( model, dic,doc,weighted_query,  ques_num, ans_tfidf)
        score_record = score_record + temp_score
        rank = rank + temp_rank

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
    model , dic, doc , qus_tfidf= get_lda_model(qus)
    query_model, query_corpus, query_dct = get_query_weight(org)

    for ques_num in tqdm(range(0,  len(ORGQ_ID), 10), ascii = True):

        #get weighted query by tf-idf weighted

        weighted_query = get_weighted_query(query_model, query_corpus,ques_num,test_nan(OrgQBody[ques_num ]) + test_nan(OrgQSubject[ques_num ]), query_dct)

        #get score, rank

        temp_score , temp_rank = get_similarity_score( model, dic,doc,weighted_query,  ques_num, qus_tfidf)
        score_record = score_record + temp_score
        rank = rank + temp_rank

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
    model , dic, doc , ans_tfidf= get_lda_model(ans)
    for ques_num in tqdm(range(0,  len(ORGQ_ID), 10), ascii = True):
        #get weighted query by tf-idf weighted

        weighted_query = get_weighted_query(query_model, query_corpus,ques_num, test_nan(OrgQBody[ques_num ]) + test_nan(OrgQSubject[ques_num ]), query_dct)

        #get score, rank

        temp_score , temp_rank = get_similarity_score( model, dic,doc,weighted_query,  ques_num, ans_tfidf)
        score_record = score_record + temp_score
        rank = rank + temp_rank


    subtaskC = pd.DataFrame({'ORGQ_ID' : ORGQ_ID, 'RELC_ID' : RELC_ID, 'RELC_RELEVANCE2RELQ' : RELC_RELEVANCE2RELQ, 'bm25' : score_record, 'rank' : rank})
    subtaskC = subtaskC[['ORGQ_ID', 'RELC_ID', 'rank', 'bm25', 'RELC_RELEVANCE2RELQ']]

    subtaskC.to_csv(os.path.join(cur_dir, outputname), header=None, index=None, sep=' ', mode='a')




    #save output
def excu_lda():
    data = pd.read_csv(os.path.join(cur_dir, "test_use.csv"), engine='python')
    print("---lda subtaskA----")
    subtaskA(data, "lda_subtaskA.pred")
    print("---lda subtaskB----")
    subtaskB(data, "lda_subtaskB.pred")
    print("---lda subtaskC----")
    subtaskC(data, "lda_subtaskC.pred")
