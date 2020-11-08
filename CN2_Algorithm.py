# =============================================================================
# HOMEWORK 3 - RULE-BASED LEARNING
# CN2 ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# For this project, the only thing that we will need to import is the "Orange" library.

from Orange.data import Table
from Orange.classification.rules import CN2Learner, CN2UnorderedLearner, LaplaceAccuracyEvaluator
from Orange.evaluation import CrossValidation, CA, Precision, Recall, F1

# Load 'wine' dataset
# =============================================================================

wineData = Table.from_file(filename='wine.csv', sheet='wine')

# Define the learner that will be trained with the data.
# Try two different learners: an '(Ordered) Learner' and an 'UnorderedLearner'.
# =============================================================================

learner = CN2Learner()
unordered_learner = CN2UnorderedLearner()

# At this step we shall configure the parameters of our learner.
# We can set the evaluator/heuristic ('Entropy', 'Laplace' or 'WRAcc'),
# 'beam_width' (in the range of 3-10), 'min_covered_examples' (start from 7-8 and make your way up),
# and 'max_rule_length' (usual values are in the range of 2-5).
# They are located deep inside the 'learner', within the 'rule_finder' class.
# Note: for the evaluator, set it using one of the Evaluator classes in classification.rules
# =============================================================================

learner.rule_finder.general_validator.beam_width = 8
learner.rule_finder.general_validator.min_covered_examples = 15
learner.rule_finder.general_validator.max_rule_length = 5

# By default evaulator is Entropy
learner.rule_finder.quality_evaluator = LaplaceAccuracyEvaluator()

unordered_learner.rule_finder.general_validator.beam_width = 5
unordered_learner.rule_finder.general_validator.min_covered_examples = 7
unordered_learner.rule_finder.general_validator.max_rule_length = 3

# =============================================================================

# We want to test our model now. The CrossValidation() function will do all the
# work in this case, which includes splitting the whole dataset into train and test subsets,
# then train the model, and produce results.
# So, simply initialize the CrossValidation() object from the 'testing' library
# and call it with input arguments 1) the dataset and 2) the learner.
# Note that the 'learner' argument should be in array form, i.e. '[learner]'.

cv = CrossValidation()
res = cv(data=wineData, learners=[learner])

cv2 = CrossValidation(k=10)
res2 = cv2(data=wineData, learners=[unordered_learner])

# As for the required metrics, you can get them using the 'evaluation.scoring' library.
# The 'average' parameter of each metric is used while measuring scores to perform
# a type of averaging on the data. DON'T WORRY MUCH ABOUT THAT JUST YET (AGAIN). USE EITHER
# 'MICRO' OR 'MACRO' (preferably 'macro', at least for final results).
# =============================================================================


# # ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print("Accuracy for ordered learner:", CA(res))
print("Precision for ordered learner:", Precision(res, average="macro"))
print("Recall for ordered learner:", Recall(res, average="macro"))
print("F1 for ordered learner:", F1(res, average="macro"))

print('------------------------------------------------------------------')

print("Accuracy for unordered learner: :", CA(res2))
print("Precision for unordered learner: :", Precision(res2, average="macro"))
print("Recall for unordered learner: :", Recall(res2, average="macro"))
print("F1 for unordered learner: :", F1(res2, average="macro"))

# =============================================================================

# Ok, now let's train our learner manually to see how it can classify our data
# using rules.You just want to feed it some data- nothing else.
# =============================================================================

classifier = learner(data=wineData)

unordered_classifier = unordered_learner(data=wineData)

# =============================================================================


# Now we can print the derived rules. To do that, we need to iterate through
# the 'rule_list' of our classifier.

print('------------------------------------------------------------------')

for rule in classifier.rule_list:
    print(rule)

print('------------------------------------------------------------------')

for rule in unordered_classifier.rule_list:
    print(rule)


