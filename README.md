# Duolingo's Half-Life Regression for Spaced Repetition
Replication of Duolingo's spaced repetition Half Life Regression based mostly on their ACL16 paper.

So far I've replicated the logistic regression baseline, but numeric errors impede HLR to work correctly with some batch sizes, and automatic differentiation seems to not be producing the correct gradients. My next step will be to implement the backward phase for hlr manually.

The final results can be seen in the file results.ods, and are averaged over 5 different runs. Note that the values replicated are different from the ones seen in the paper due to the fact that I use mini-batches and train for 10 epochs shuffling the data instead of just one pass through the dataset without shuffling, and due to the fact that pytorch's automatic differentiation may be giving out different gradients than the expected ones.
