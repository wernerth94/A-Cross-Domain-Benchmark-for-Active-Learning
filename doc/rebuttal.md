# Reviewer iCXF
rating: 7, confidence 3
```
Thank you for your review...
```
The results on the synthetic benchmarks are unsurprising.
The algorithm for the greedy oracle has no clear guarantees, the design choices are unclear and the description is hard to follow.
The evaluation is done with relatively small models and datasets and it is unclear whether it translates to larger ones.

# Reviewer 3W2R
rating: 7, confidence 3
```
Thank you for your review...
```
However this work should probably cite and discuss Zhang et al.
```
We will add Zhang et al to Table 1 and our Related Work
```

The "oracle" needs to be explained earlier. It is discussed and referenced a lot, but you don't actually describe what it is until section 6. I was very confused, since typically "oracle" might also refer to an annotator. Things made sense at section 6, but a preview of what exactly the oracle does should be discussed as soon as possible. Looking again, it is discussed at line 53 but I actually found this paragraph very vague.
```
The entire section explaining the oracle will be revised for the camera ready version, apart from introducing the concept earlier in the paper, we aim to improve the understanding of readers without prior knowledge.
```

what is "sngl" in Table 1? Single point sampling per round?
```
Yes, we will improve the clarity in Table 1
```

what is 9(14) in Table 1 in the last row?
```
We have 9 datasets in our benchmark, 5 of whichhave a pre-encoded version (excluding text and synthetic) , which brings the total number of experiments to 14.
```

not sure what i\in means really after line 93
```
It simply indexes all points in the unlabeled set.
We will improve the clarity here as well
```

I think Figure 1 needs better descriptions and a better legend. I think I get the main point but it's confusing. E.g., "True mean" is vague, what is that the true mean of? Purple curve doesn't even appear on the legend.
```
Similar to the explanations about the oracle, we aim to revise this section to improve the understanding of readers without prior knowledge.
```

the choice of validation on the entire dataset needs more discussion (line 194). This is a huge criticism of active learning research, to choose parameters based on a full validation set. I understand the argument for lower variance in research evaluation, but I don't think the justification here is sufficient. In particular, Figure 1 argues that the high variance in research results is a problem. Why mask it, with an unrealistic validation? Won't that make things worse?
```
@Max 

```

# Reviewer x7yL
rating: 4, confidence 5

```
Thank you for your review...
```
Although the author states that the proposed benchmark is the first one that contains multiple domains, however, in each domain, many related empirical studies focus on specific domain, e.g., [r1], [r2], [r3], which weakens the contribution of this paper.
```
We will include [r1] into Table 1 and our related work ([r2] and [r3] do not provide experiments on their own).
AL benchmarks are still riddled with comparability issues, because their choice of datasets, domains and models ([r2] Sec. 4.1 and [r3] Tab. 1) often is completely disjoint.
In our opinion, we need a benchmark that spans all domains and the most commonly used datasets and models to fix this problem.
As described in our Section 5, we selected our datasets for maximum intersection with previous works. The same is true for our model selection, which either uses common models like ResNet18 or trivial to reproduce models like MLPs.
[r2]: Zhang, Zhisong, Emma Strubell, and Eduard Hovy. "A survey of active learning for natural language processing." arXiv preprint arXiv:2210.10109 (2022).
[r3] Schröder, Christopher, and Andreas Niekler. "A survey of active learning for text classification using deep neural networks." arXiv preprint arXiv:2008.07267 (2020).
```

The insights (e.g., the importance of multiple runs) build on existing knowledge in the active learning field.
```
Even though it is common knowledge that repeating an experiment enough times is crucial for ML research, in the field of Active Learning we still observe many authors not doing so. 
To the best of our knowledge, we are the first to provide a first insight into exactly how often an AL experiment should be repeated in order to generate reliable results.
The same is true for experimenting on different domains. Everyone knows that algorithms should be tested in as many domains as possible, but few actually apply that knowledge.
```

This paper takes too much space on the greedy oracle algorithm and seeding strategies, which may be complex for some readers to fully get the meaning.
```
We will revise the entire section about our oracle algorithm and work with more collegues from outside the paper to ensure that it can be understood without prior knowledge.
Based on the suggestion of another reviewer, we will also generally extend our explanations about the oracle, placing a high level introduction of the algorithm early in the paper to introduce the reader to the concept.
```

In the main manuscript, the author takes too much space to discuss why they need to have such experimental settings and how they set the experiments. The discussions of the experimental results are weak. Maybe this paper is more suitable to submit to a journal.
```
As you have mentioned in your review, there exists a fair number of benchmarks for AL already.
Each of those specialize on 1-2 domains and place high emphasis on their results.
However, apart from [1], each benchmark has the experimental setup and technical details on low priority, leading to a situation, where everyone is doing their own framework in isolation and an informed discussion about best practices is very hard.
We hope (in combination with [1]) to open the door for such discussion in the area of AL.
Nonetheless, we will extend discussion of our results. 
Based on the comment from another reviewer, we are planning to study our results from the dimension of class imbalance, which currently is missing in our work. 
[1]: Ji, Yilin, et al. "Randomness is the root of all evil: more reliable evaluation of deep active learning." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2023.
```

"In semi-supervised learning, which is to train a fixed encoder-model, pre-encode the datasets and then only train a single linear layer as classifier." It can be computationally efficient, but it will not fully exploit the strengths of deep learning in an active learning scenario. A more typical and potentially more effective approach in deep active learning involves training the feature extractor alongside the classifier to ensure that the representations are continuously adapted to the newly labeled data.
```
As described in Section 5.2 we are not interested in beating the SOTA performance on any dataset, but to provide a fast and low-variance framework.
The same idea applies to our choice of semi-supervised learning.
For that reason we opted for pre-encoded datasets instead of more sophisticated approaches.
Finally, the authors of [4] provide good evidence, that the ranking of algorithms does not change when the semi-supervised training changes (at least for the image domain).
[4]: Zhang, Jifan, et al. "LabelBench: A Comprehensive Framework for Benchmarking Adaptive Label-Efficient Learning." Journal of Data-centric Machine Learning Research (2024).
```

# Reviewer c7pZ
rating: 5, confidence 5
```
Thank you for your review...
```

Include more related benchmarks. For example [1] studies extensively the behavior of active learning in tabular datasets. LabelBench [2] has also proposed using embedded features and semi-supervised learning as part of their benchmark. While the authors in this paper conjecture "a well-performing method in our benchmark will also generalize well to larger datasets and classifiers", LabelBench has already demonstrated this to be true. In addition, margin sampling performing well on tabular datasets is also found in [1].
```
[1] and [2] will be integrated into Table 1 and our related work section.
Based on the comment of another reviewer we aim to extend our discussion section. 
This will also include a more in-depth comparison of our results with other benchmarks like [1].
```

The proposed metric of average ranking may not be a convenient/intuitive metric for benchmarks. Specifically, whenever a new algorithm is introduced, the scores of every algorithm will change. Moreover, practitioners are generally interested in either accuracy or label-efficiency (number of labels needed to reach a certain accuracy). The adopted metric does not capture any of these quantities directly. In some cases, entropy, badge and margin may all perform very similarly in accuracy/label-efficiency but remains stable in their relative ranking. The metric in this paper would significantly penalize the algorithms that perform relatively worse, which does not give an accurate quantification in the actual performance of the algorithms themselves.
```
We fully agree with the reviewer that accuracy scores need to be reported for any classification problem that is not severely imbalanced.
We have all accuracy scores in Appendix K, including standard deviations and the amount of wins per algorithm.
We would like to point out that our benchmark aggregrates results from different datasets, from which we cannot simply average the accuracies to display them in a table, etc.
For a fair comparison, we rely on the paired-t-test, which the Critical-Difference-Diagrams conveniently provide.
Lastly, we have been unable to include a second results table besides Table 3 within the page limit.
```

An interesting finding in LabelBench is imbalanced active learning algorithms like GALAXY performs significantly better on imbalanced datasets. Since most of the language and tabular datasets are imbalanced, it would be interesting to see how such algorithms perform. The conclusion that "entropy sampling" or "margin sampling" is the best may not be entirely accurate.
```
We omitted information about class imbalance for our datasets, as we did not focus on that aspect.
However, both our text, and 2 out of 3 tabular datasets are imbalanced by nature. Implementing GALAXY and discussing results along this dimension as well is a great suggestion. 
Since testing GALAXY on our entire benchmark is a significant computational effort, we only provide preliminary experiments on the TopV2 dataset.
```
