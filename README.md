# IRIE-term-project2018: Community Question Answering

The question-answering task contained three subtasks: question-question similarity, question-comment similarity, and question and external comment similarity. A new question and corresponding collection of questions and comments were given, and the goal was to produce the ranking of the relevance of questions or comments to the given question. To cope with the task, the BM25 and LDA approaches were used to rerank the questions and comments in the given collection, and MAP (mean average precision) was applied to evaluate the performance. The corpus, BM25, and LDA model were built by genism library.


# Data: SemEval-2017


 * Train:
    * SemEval2015-Task3-CQA-QL-dev-reformatted-excluding-2016-questions-cleansed.xml
    * SemEval2015-Task3-CQA-QL-test-reformatted-excluding-2016-questions-cleansed.xml
    * SemEval2015-Task3-CQA-QL-train-reformatted-excluding-2016-questions-cleansed.xml
    * SemEval2016-Task3-CQA-QL-dev.xml
    * SemEval2016-Task3-CQA-QL-test.xml
    * SemEval2016-Task3-CQA-QL-train-part1.xml
    * SemEval2016-Task3-CQA-QL-train-part2.xml
 * Test: 
     * SemEval2017-task3-English-test-input.xml
 * Data Format:
    * OrgQSubject: The subject of the original question
    * OrgQBody: The main body of the question
    * Thread: A Thread consists of a potentially relevant question RelQuestion, together with 10 comments RelComment for it.
    * RelCText: The text of the comment
    * RELQ_RELEVANCE2ORGQ in RelQuestion:  relevance of the thread of this RelQuestion with respect to the OrgQuestion (**PerfectMatch**, **Relevant**, **Irrelevant**)
    * RELC_RELEVANCE2**ORGQ** in RelComment:  human assessment about whether the comment is "Good", "Bad", or "PotentiallyUseful" with respect to the OrgQuestion (**Good**, **PotentiallyUseful**, **Bad**)
    * RELC_RELEVANCE2**RELQ** in RelComment: human assessment about whether the comment is "Good", "Bad", or "PotentiallyUseful" with respect to the RelQuestion (**Good**, **PotentiallyUseful**, **Bad**)


# Task Description
 1. English subtask A (Question-Comment Similarity): 
Given **a question** and the first **10 comments** in its question thread,**rerank** these 10 comments according to their relevance with respect to the question.
 2. English subtask B (Question-Question Similarity):
Given **a new question** (aka original question) and the set of the first **10 related questions** (retrieved by a search engine), **rerank** the related questions according to their similarity with the original question.
3. English subtask C (Question-External Comment Similarity):
Given **a new question** (aka the original question) and the set of the first 10 related questions (retrieved by a search engine), each associated with its first 10 comments appearing in its thread, **rerank the 100 comments** (10 questions x 10 comments) according to their relevance with respect to the original question.


# Data preprocessing:

First, the entities including IDs, OrgQSubject, OrgQBody, RelQSubject, RelQBody, and RelCtext were fetched from data. Second, preprocessing followed the five steps: 
1. Noise removal
2. Contractions and punctuation replacement
3. Word tokenization
4. Lowercases and numbers conversion
5. Non-ASCII characters and stopwords removal

# Model

## Okapi BM25

Okapi BM25 developed in 1970s is a TF-IDF-like ranking function used to computed relevance of document to the quires and it is still considered as a baseline for evaluating new ranking functions. In this project, the parameter b, K1 in the BM25 formula were given different values for searching good ranking scores.


![BM25 formula](https://github.com/lilinmail0523/IRIE-term-project2018/blob/master/image/BM25.png)

q<sub>i</sub> was ith term query. f<sub>i,j</sub> told how many times i<sup>th</sup> term in query which occurs in document j. K<sub>1</sub> and b were parameters that depend on cases.

* PS: Because the K<sub>1</sub> and b would be changed in BM25, the [genism BM25](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/summarization/bm25.py) was taken here (name: bm25_util.py) for further processes.  

![BM25 result K1](https://github.com/lilinmail0523/IRIE-term-project2018/blob/master/image/BM25ResultK1.png)

![BM25 result b](https://github.com/lilinmail0523/IRIE-term-project2018/blob/master/image/BM25Resultb.png)

Results showed that the smaller the k1 and b, the better the MAP. The parameter K1 controls the term frequency saturation rate, and K1 could be larger when lots of different terms appear in a work, while there was less likelihood of high saturation speeds in this task. The parameter b controls the influence of document length normalization. The results presented the cases with lower b had higher performance because documents were short and highly specific in a certain field.


## LDA

LDA is a kind of topic model which is applied to estimate the relevance between the queries and documents by unsupervised learning of topic distribution. A document is represented by a vector of the probability distribution of topics, and a topic consists of the probability distribution of words. In this project, because the length of questions and comments were too short, the word topics were used rather than document topics, and the topics of questions and comments were accumulated by word topics multiplied by its frequency or tf-idf weight. Finally, the similarity was evaluated by cosine similarity and L2 distance. 

Average: term topic * term **frequency** in documents/queries and sum up to obtain documents/queries topic scores.

tf-idf: term topic * term **tf-idf weight** in documents/queries and sum up to obtain documents/queries topic scores.

![LDA result b](https://github.com/lilinmail0523/IRIE-term-project2018/blob/master/image/LDAResult.png)

cos: cosine similarity/ l2: L2 distance

Results showed that term tf-idf weight outperformed term frequency because tf-idf model contained idf (inverse document frequency) to reflect the importantance of a word to a document/query. When it came to similarity ranking, cosine similarity outperformed L2 norm because the direction contained information like "style" or "sentiment" in the vectors.

# reference

- [term.pdf](https://github.com/lilinmail0523/IRIE-term-project2018/blob/master/term.pdf)
- Community Question Answering: [SemEval-2017Task3: CommunityQuestionAnswering](https://www.aclweb.org/anthology/S17-2003.pdf) [[Website](http://alt.qcri.org/semeval2017/task3/)]
- Data processing: [Text Data Preprocessing: A Walkthrough in Python](https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html)
- [Practical BM25 - Part 3: Considerations for Picking b and k1 in Elasticsearch](https://www.elastic.co/blog/practical-bm25-part-3-considerations-for-picking-b-and-k1-in-elasticsearch)
