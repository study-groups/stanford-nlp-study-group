# Lecture 2: Word Vector Representations: word2vec
[Lecture 2 Stanford NLP on word2vec](https://www.youtube.com/watch?v=ERibwqs9p38)
  - [slides](https://github.com/study-groups/stanford-nlp-study-group/blob/master/web2017/lectures/cs224n-2017-lecture2.pdf)


# Lecture Plan
- Lecture plan: [2:28](https://www.youtube.com/watch?v=ERibwqs9p38&t=2m28s)
- Word meaning: [4:20](https://www.youtube.com/watch?v=ERibwqs9p38&t=4m20s)
- Word2vec: [18:30](https://www.youtube.com/watch?v=ERibwqs9p38&t=18m30s)
- Reseach highlight: [42:30](https://www.youtube.com/watch?v=ERibwqs9p38&t=42m30s)
- Word2vect objective function gradients: [47:20](https://www.youtube.com/watch?v=ERibwqs9p38&t=47m20s)
- Optimization refresher: [1:12:00](https://www.youtube.com/watch?v=ERibwqs9p38&t=1h12m00s)

# Highlights
## Word meaning
- Distributed representation of meaning [10:54](https://www.youtube.com/watch?v=ERibwqs9p38&t=10m54s)
  - one-hot encoding 
  - localist, no inherent notion between words
  - denotational. 'right here' 
 
- Distributional Similarity [15:00](https://www.youtube.com/watch?v=ERibwqs9p38&t=15m0s)
  - Wittgenstein's "use theory of meaning"
  - consider a word's use in context
  - You shal know a word by the company it keeps." J.R. Firth
  - not the same as distributed representation of meaning
  - create a dense vector that will dot-product nicely to suggest similarity

## Word2Vec
- A neural probabilistic language model -Bengio, 2003
- Main idea of word2vec [23:45](https://www.youtube.com/watch?v=ERibwqs9p38&t=23m45s)
  - predict between every word and its context of words
- Two algorithms
  - **skip-grams (SG):** predict context words given target
  - continuous Bag of Words (CBOW)
- Two training methods (not covered)
  - Hierarchical softmax
  - Negative sampling
  - Consider Naive softmax method instead
- Skip-gram prediction [25:00](https://www.youtube.com/watch?v=ERibwqs9p38&t=10m54s)
  - Only one probability distribution for all output context words
- Detains of word2vec [29:00](https://www.youtube.com/watch?v=ERibwqs9p38&t=29m00s)
  - For each word t = 1 to T, predict surrounding words in a window of radius m. 
  - **Objective Function:** Maximize the probability of any context word given the current
  center word.
    - Product of all probabilities over all words results in a prodcut of products objective function. 
    - Take the log to turn products into sums. 
    - change maximization to minimization by adding negaitve sign
    - We are taking the **Negative Log Likelihood** (cross entropy loss)
    - we are keeping window length out of the eq'n to minimize
    - With one-hot w_(t+j) we are just predicting the word that occured 
      - the only term left is the negative probability of the true class
- How doe minimize the negative log likelihood? [31:30](https://www.youtube.com/watch?v=ERibwqs9p38&t=29m00s)
  - p(w_(t+j) | w_t )
  - Dot products
  - softmax turns numbers into probability distributions
  - exponentiate to make positive (negative exponents are less than 1 but not negative)
  - p = exp( u_o dot v_c) / sum exp(u_w dot v_c)
    - o = outside word index
    - c = center word index
    - **v_c** and **u_o** are "center" and "outside" vecturs of indicies c and o
  - Softmax using word c to obtain probability of word o
  - Iterate oer w=1...W: u_w dot v  (v is center, u is outer)
  - One word gets 2 representations: one when its a center, one when its an outer
  - Context words can fall anywhere- order or position not considered
- Skip-gram [39:00](https://www.youtube.com/watch?v=ERibwqs9p38&t=39m00s)
  - w_t = center-word one-hot vector vx1 
  - W = word embedding matrix as a representations of center words (each column is a word) 
  - w_t * W => select out the column of the Matrix which is the center word
  - w_t * W = v_c
  - v_c * ContextWordMatrix =  where CWM is  (v x d) and v_c is (d x 1)
  - Softmax yields "Probability of each word appearing in the context given a certain word is the center word."
- Compute all vector gradients! [41:47](https://www.youtube.com/watch?v=ERibwqs9p38&t=41m47s)
  - Theta is the set of **all** parameters in one long vector
  - d dimensional vector with V many words
    - `[v_aardvark ... v_zebra, u_aardvark ... u_ u_zebra]^T  element of R^(2*d*V)`

## Research [42:30](https://www.youtube.com/watch?v=ERibwqs9p38&t=42m30s)
 - weighted bag of words + remove some special direction
 - vectorize via sentences
 - apply PCA
 - probabilist interpretation
 - smoothing function and position significant
 
## Word2vect objective function gradients: [47:20](https://www.youtube.com/watch?v=ERibwqs9p38&t=47m20s)
- partial derivative w.r.t. v_c: [52:20](https://www.youtube.com/watch?v=ERibwqs9p38&t=52m20s)
- good to work out vector derivatives
- chain rule: [55:00](https://www.youtube.com/watch?v=ERibwqs9p38&t=55m00s)
  - back-prop is chain rule plus memoization (store intermediate values)
  - result: [1:04:00](https://www.youtube.com/watch?v=ERibwqs9p38&t=1h04m00s)
    - observed - expectation
    - u_o      - sum_over_x [ p( x | c) * u_x]
    - output context - probability of every possible word appearing in the context * take that much of u_x expectation vector 
    -  Said another way: average over all of the context vectors (u_x) weighted by their likelyhood of their occurance p(x|c)
- must repeat for the context vectors

## Optimization refresher: [1:12:00](https://www.youtube.com/watch?v=ERibwqs9p38&t=1h12m00s)
- Derivative of the objective function
- Vanilla (batch) gradient descent:   [1:14:30](https://www.youtube.com/watch?v=ERibwqs9p38&t=1h14m30s)
  - would need to compute gradients for all windows
  - theta_new = theta_old - alpha * grad{J(Theta)} (gradient is with respect to theta_j
    - theta_grad = evaluate_gradient(J,**corpus**,theta)
    - theta = theta - alpha * theta_grad

- Stochastic Gradient Descent: [1:16:00](https://www.youtube.com/watch?v=ERibwqs9p38&t=1h16m00s)
  - We don't consider every window
  - just take one center word and calculate the gradient
  - very bad estimate at one position
  - sample_window(corpus)
    - theta_grad = evaluate_gradient(J,**window**,theta)
    - theta = theta - alpha * theta_grad

# Suggested reading
  - http://web.stanford.edu/class/cs224n/syllabus.html
  - [skip-gram-tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/#)
  - [distributed-representation-paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  -[Efficient Estimation of Word Representations in Vector Space by Tomas Mikolov](https://arxiv.org/pdf/1301.3781.pdf)

# Assignment info
  - [HW#1](http://web.stanford.edu/class/cs224n/assignment1/index.html)
  - [HW#1-solutions](http://web.stanford.edu/class/cs224n/assignment1/assignment1-solution.pdf)

# References
- [Stanford Encyclopedia of Philosophy on Wittgenstein](https://plato.stanford.edu/entries/wittgenstein/)
- [Ludwig Wittgenstein](https://en.wikipedia.org/wiki/Ludwig_Wittgenstein)
- [how_does_word2vec_work](http://www.1-4-5.net/~dmm/ml/how_does_word2vec_work.pdf)
- [tensor flow example](https://www.tensorflow.org/tutorials/representation/word2vec)
- [Word2Vec Giant Leap in Language Processing](https://medium.com/explore-artificial-intelligence/word2vec-a-baby-step-in-deep-learning-but-a-giant-leap-towards-natural-language-processing-40fe4e8602ba)
