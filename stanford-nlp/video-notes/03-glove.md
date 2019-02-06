# Lecture 3: GloVe: Global Vectors for Word Representation
[Lecture 3 Stanford NLP on GloVe](https://www.youtube.com/watch?v=ASn7ExxLZws&index=4&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&t)
  - [slides](http://web.stanford.edu/class/cs224n/lectures/lecture3.pdf)

# Lecture Plan
- Lecture plan: [2:28](https://youtu.be/ASn7ExxLZws?t=2m28s)
  - Finish word2vec: [3:07](https://youtu.be/ASn7ExxLZws?t=3m7s)
  - What does word2vec capture? [25:05](https://youtu.be/ASn7ExxLZws?t=25m05s)
  - Break: Linear Algebraic Structure of Word Senses [45:45](https://youtu.be/ASn7ExxLZws?t=45m45s)
  - How could we capture this essence more effectively? [31:14](https://youtu.be/ASn7ExxLZws?t=31m14s)
  - How can we analyze word vectors?
    - Evaluation [50:35](https://youtu.be/ASn7ExxLZws?t=50m35s)
   

# Highlights

## Skip-gram and negaitve sampling
- A neural probabilistic language model -Bengio, 2003
- Main idea of word2vec [11:13](https://youtu.be/ASn7ExxLZws?t=11m13s)
  - Basic idea: log[sigma(outerwords dot centerword)  divided by sum of log[ sigma(random samplings dot centerword)]
  - Unigram matrix to de-emphasis stop words

## Continuous Bag of Words
- word2vec improves object function by putting similar words nearby in space
- Main idea of Continuous Bag of Words [17:50](https://youtu.be/ASn7ExxLZws?t=17m50s)
  - Add the nearby words the dot-product against centerword
  
## Summary of word2vec algorithm
- Go thru corpus find all co-occuring: [19:37](https://youtu.be/ASn7ExxLZws?t=19m37s)
- Capture the entire count of co-occurance
- Looking arond the document "this word appears with enitre document"  -> Laten Semantic Analysis [21:40](https://youtu.be/ASn7ExxLZws?t=21m40s)

## Simple example of Windw based co-ocurance matrix
- Buiding a co-occurance matrix: [24:18](https://youtu.be/ASn7ExxLZws?t=24m18s)

## How do we reduce the large co-occurance matrix?
- SVD: [27:20](https://youtu.be/ASn7ExxLZws?t=26m30s)

## Hybrid solution: GloVe
- Count based v. direct perdiction: [35:54](https://youtu.be/ASn7ExxLZws?t=35m54s)
- GloVe cost function, J(theta):[38:00](https://youtu.be/ASn7ExxLZws?t=38m00s)

## Word vector analogies
- Intrinsic word vector evaluaion: [54:30](https://youtu.be/ASn7ExxLZws?t=54m30s)
=

# Suggested reading
  - http://web.stanford.edu/class/cs224n/syllabus.html
  - [skip-gram-tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/#)
  - [distributed-representation-paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  -[Efficient Estimation of Word Representations in Vector Space by Tomas Mikolov](https://arxiv.org/pdf/1301.3781.pdf)

# Assignment info
  - [HW#1](http://web.stanford.edu/class/cs224n/assignment1/index.html)
  - [HW#1-solutions](http://web.stanford.edu/class/cs224n/assignment1/assignment1-solution.pdf)

# References
- [Global Vectors for Word Representatino by Chtid Manning et al](https://nlp.stanford.edu/pubs/glove.pdf)
- [distributed-representations-of-words-and-phrases-and-their-compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)

