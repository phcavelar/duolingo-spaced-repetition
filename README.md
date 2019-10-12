# Duolingo's Half-Life Regression for Spaced Repetition
Replication of Duolingo's spaced repetition Half Life Regression based mostly on their ACL16 paper.

So far I've replicated their model and the logistic regression baseline, but numeric errors impede HLR to work correctly with some batch sizes.

The final results can be seen in the file results.ods, and are averaged over 5 different runs. Note that the values replicated are different from the ones seen in the paper due to the fact that I use mini-batches and train for 10 epochs shuffling the data instead of just one pass through the dataset without shuffling.
