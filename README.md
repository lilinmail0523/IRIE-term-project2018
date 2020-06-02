# IRIE-term-project2018: Community Question Answering


Community Question Answering (CQA) provides the information to users more flexibility. The platform such as Stack Overflow and Quora offer a place that anyone can post and answer a question. Some of anwers are helpful for users, and some of them are unrelated to the question. It may take much time for users to find the correct answer. So the task can help automate the process of finding good answers to the certain question.

For further information of Community Question Answering:
 - [SemEval-2017Task3: CommunityQuestionAnswering](https://www.aclweb.org/anthology/S17-2003.pdf) [[Website](http://alt.qcri.org/semeval2017/task3/)]

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

For futhur information about the structure of data: [term.pdf](https://github.com/lilinmail0523/IRIE-term-project2018/blob/master/term.pdf)

# Task Description
 1. English subtask A (Question-Comment Similarity): 
Given **a question** and the first **10 comments** in its question thread,**rerank** these 10 comments according to their relevance with respect to the question.
 2. English subtask B (Question-Question Similarity):
Given **a new question** (aka original question) and the set of the first **10 related questions** (retrieved by a search engine), **rerank** the related questions according to their similarity with the original question.
3. English subtask C (Question-External Comment Similarity):
Given **a new question** (aka the original question) and the set of the first 10 related questions (retrieved by a search engine), each associated with its first 10 comments appearing in its thread, **rerank the 100 comments** (10 questions x 10 comments) according to their relevance with respect to the original question.

# Evaluation
 *  We use MAP (mean average precision) to evaluate the performance

For further information of the project: [term.pdf](https://github.com/lilinmail0523/IRIE-term-project2018/blob/master/term.pdf)


# Data preprocessing:
In this project, the test data including IDs, OrgQSubject, OrgQBody, RelQSubject, RelQBody and RelCtext were used for furthur processes. Data preprocessing consists of the following steps:
1. Noise removal: Extracting data from xml files
2. Contractions and punctuation replacement: removing punctuation
3. Word tokenization: spliting the sentences into words
4. Lowercases and numbers conversion
5. non-ASCII characters and stopwords removal (by nltk stopwords corpus)

# Model
CO<sub>2</sub>

## BM25

![BM25 formula](https://github.com/lilinmail0523/IRIE-term-project2018/blob/master/image/BM25.png)

We'll take a look at BM25 formula. q<sub>i</sub> is ith term query. f<sub>i,j</sub> tells how many times i<sup>th</sup> term in query which occurs in document j. K<sub>1</sub> and b are parameters that depend on cases. With BM25 scores, we can give documents a rank.

* PS: Because the K<sub>1</sub> and b would be changed in BM25, the [genism BM25](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/summarization/bm25.py) was taken here (name: bm25_util.py) for furthur processes.  

![BM25 result K1](https://github.com/lilinmail0523/IRIE-term-project2018/blob/master/image/BM25ResultK1.png)

![BM25 result b](https://github.com/lilinmail0523/IRIE-term-project2018/blob/master/image/BM25Resultb.png)

 In this project, results told that the smaller the k1 and b, the better the MAP.


## LDA
In LDA, each word is viewed as a mixture of various topics. We can assign each word a set of topics probability via LDA. To acquire document topics vectors, we can use word topics to deal with it. Finding a proper way converting word topics to documents/quires topics is viewed as hard work. In this project, we apply two approach (average, tf-idf) to do so.

Average: term topic * term **frequency** in documents/queries and sum up to obtain documents/queries topic scores.
tf-idf: term topic * term **tf-idf weight** in documents/queries and sum up to obtain documents/queries topic scores.

![LDA result b](https://github.com/lilinmail0523/IRIE-term-project2018/blob/master/image/LDAResult.png)

cos: cosine similarity/ l2: L2 distance

Not surprisingly, term tf-idf weight outperform term frequency. Because tf-idf model intend to reflect how important a word is to a document/query. For similarity calculation, cosine similarity is a common tool to compare two vectors. L2 distance, known as Euclidean distance, results in poor performance.  
