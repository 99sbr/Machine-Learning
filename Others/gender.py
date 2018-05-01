import nltk
from nltk.corpus import names
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +[(name, 'female') for name in names.words('female.txt')])
import random
def gender_features(word):
	return {'last_letter': word[-1] , 'Length': len(word)}
random.shuffle(labeled_names)
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
from nltk.classify import apply_features
train_set = apply_features(gender_features, labeled_names[1500:])
test_set = apply_features(gender_features, labeled_names[500:1500])
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
print("The person is %s " % classifier.classify(gender_features('')))