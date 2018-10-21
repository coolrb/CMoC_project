'''
data_cleaning.py

This file parses through the Enron email data csv file and returns a shortened csv file that
gets rid of all the reply/forward emails as well as emails with non-extractable words in the subjects
'''
import pandas as pd
import nltk

'''
Parses through the email csv and returns a pandas dataframe
'''

stoplist = ['a', 'an', 'the', 'and', 'or', 'of', 'for', 'to', 'in', 'from', 'not', 'but', 'up']
symbols = '!@#$%^&*()_+-=;:",./<>?\\'

def parse_csv(file):
	df = pd.read_csv(file, sep = "\n", header = [''])

	for index, row in df.iterrows():
		subject = row['subject'].lower()

		if checkReForward(subject):
			continue

		subject = subject.replace(symbols, "")
		subject = stopwords(subject)

		line = row['text'].lower()
		line = line.replace(symbols, "")
		line = stopwords(line)

		if not checkNonExtract(subject, line):
			# change them in the data frame
			df[index]['text'] = line
			df[index]['subject'] = subject
		else:
			df.drop([index])

	return df.reset_index()

def checkReForward(subject):
	if "re:" in s or "fwd:" in s:
		return True

def stopwords(line):
	for stop in stopwords:
		line = line.replace(stop, "")

def checkNonExtract(subject, txt):
	words = subject.split()
	txt = txt.split()

	for word in words:
		if word not in txt:
			return False
	return True

def main():
	csv = parse_csv("")


main()