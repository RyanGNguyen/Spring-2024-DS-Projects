import argparse
from itertools import groupby

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


class SegmentClassifier:
    def train(self, trainX, trainY):
        self.clf = GradientBoostingClassifier() # TODO: experiment with different models
        X = [self.extract_features(x) for x in trainX]
        self.clf.fit(X, trainY)

    def extract_features(self, text):
        words = text.split()
        features = [  # TODO: add features here
            len(text),
            len(text.strip()),
            len(words),
            text.count(' '),
            text.count('.'),
            text.count(','),
            text.count(';'),
            text.count(':'),
            text.count('?'),
            text.count('!'),
            text.count('\''),
            text.count('"'),
            text.count('\\'),
            text.count('/'),
            text.count('+'),
            text.count('-'),
            text.count('*'),
            text.count('='),
            text.count('('),
            text.count(')'),
            text.count('['),
            text.count(']'),
            text.count('{'),
            text.count('}'),
            text.count('<'),
            text.count('>'),
            text.count('@'),
            text.count('#'),
            text.count('$'),
            text.count('%'),
            text.count('&'),
            text.count('_'),
            text.count('^'),
            text.count('|'),
            text.count('~'),
            sum(1 if w.isprintable() else 0 for w in words),
            sum(1 if w.isascii() else 0 for w in words),
            sum(1 if w.isalnum() else 0 for w in words),
            sum(1 if w.isalpha() else 0 for w in words),
            sum(1 if w.isupper() else 0 for w in words),
            sum(1 if w.islower() else 0 for w in words),
            sum(1 if w.istitle() else 0 for w in words),
            sum(1 if (w.istitle() or w.isupper()) and w.endswith(':') else 0 for w in words),  # Headers
            sum(1 if w.isnumeric() else 0 for w in words),
            sum(1 if w.isdigit() else 0 for w in words),
            sum(1 if w.isdecimal() else 0 for w in words),
        ]
        return features

    def classify(self, testX):
        X = [self.extract_features(x) for x in testX]
        return self.clf.predict(X)


def load_data(file):
    with open(file) as fin:
        X = []
        y = []
        for line in fin:
            arr = line.strip().split('\t', 1)
            if arr[0] == '#BLANK#':
                continue
            X.append(arr[1])
            y.append(arr[0])
        return X, y


def lines2segments(trainX, trainY):
    segX = []
    segY = []
    for y, group in groupby(zip(trainX, trainY), key=lambda x: x[1]):
        if y == '#BLANK#':
            continue
        x = '\n'.join(line[0].rstrip('\n') for line in group)
        segX.append(x)
        segY.append(y)
    return segX, segY


def evaluate(outputs, golds):
    correct = 0
    for h, y in zip(outputs, golds):
        if h == y:
            correct += 1
    print(f'{correct} / {len(golds)}  {correct / len(golds)}')


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--format', required=True)
    parser.add_argument('--output')
    parser.add_argument('--errors')
    parser.add_argument('--report', action='store_true')
    return parser.parse_args()


def main():
    args = parseargs()

    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)

    if args.format == 'segment':
        trainX, trainY = lines2segments(trainX, trainY)
        testX, testY = lines2segments(testX, testY)

    classifier = SegmentClassifier()
    classifier.train(trainX, trainY)
    outputs = classifier.classify(testX)

    if args.output is not None:
        with open(args.output, 'w') as fout:
            for output in outputs:
                print(output, file=fout)

    if args.errors is not None:
        with open(args.errors, 'w') as fout:
            for y, h, x in zip(testY, outputs, testX):
                if y != h:
                    print(y, h, x, sep='\t', file=fout)

    if args.report:
        print(classification_report(testY, outputs))
    else:
        evaluate(outputs, testY)


if __name__ == '__main__':
    main()