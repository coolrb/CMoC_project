'''
data_cleaning.py

This file parses through the Enron email data csv file and returns a shortened csv file that
gets rid of all the reply/forward emails as well as emails with non-extractable words in the subjects.
Thus, we can use the condensed email data for our next steps instead of running a huge chunk of data.
'''
import pandas as pd
import nltk
import os
'''
Parses through the email csv and returns a pandas dataframe and outputs a csv file
'''

def read_text():
	data = [[],[],[]]

	for file in os.listdir('enron_all_emails/'):
		f = open('enron_all_emails/' + file)
		foundBody = False

		# check if email contains forward info in email body and gets rid of that
		foundBodyFwd = False
		foundBodySub = False

		body = ""
		for line in f:
			if foundBody:
				if 'forwarded by' in line.lower():
					foundBodyFwd = True
				elif foundBodySub:
					body += line + " "
				elif foundBodyFwd:
					if 'subject:' in line.lower():
						foundBodySub = True
				else:
					body += line + " "
			else:
				if line.startswith("Subject:"):
					line = line.replace("Subject: ", "", 1)
					subject = line.strip()
				elif line.startswith('X-FileName:'):
					foundBody = True

		body = body.strip()
		if (not body) or (not subject): # check empty subject and body
			continue
		body.replace("=01&", '\'')
		body.replace("=01,", '\'')
		data[0].append(subject)
		data[1].append(body)
		data[2].append(file)

	return data

# '=01&', '=01,'

# the stopword list is quite cruicial in the process
stoplist = ['a', 'an', 'the', 'and', 'or', 'of', 'for', 'to', 'in', 'from', 'not', 'but', 'up']
symbols = '!@#$%^&*()_+-=;:",./<>?\\'
escape = ['\n', '\t']

def create_csv(d_l):

	pop_indices = []

	for i in range(len(d_l[0])):
		subject = d_l[0][i].lower()

		if checkReForward(subject):
			pop_indices.append(i)
			continue

		# subject = re.sub(escape, "", subject)

		# subject = stopwords(subject)

		line = d_l[1][i].lower()
		# line = re.sub(escape, "", line)

		for ch in symbols:
			line = line.replace(ch, ' ')
			subject = subject.replace(ch, ' ')

		# for ch in escape:
		# 	line = line.replace(ch, ' ')
		# 	subject = subject.replace(ch, ' ')
		# line = stopwords(line)

		if not checkNonExtract(subject, line):
			pop_indices.append(i)
			continue

		# update data list

		# use this instead of strip() to get rid of all empty spaces (including \n and \t)
		subject = ' '.join(subject.split())
		line = ' '.join(line.split())

		d_l[0][i] = subject
		d_l[1][i] = line

	# needs simplifying
	print("Number of reply/forward or non-extractable emails: " + str(len(pop_indices)))
	print("Number of emails after basic cleaning, such as getting rid of ones with empty subject/body: " + str(len(d_l[0])))

	subject_dl = [d_l[0][m] for m in range(len(d_l[0])) if m not in pop_indices]
	body_dl = [d_l[1][n] for n in range(len(d_l[1])) if n not in pop_indices]
	files_dl = [d_l[2][k] for k in range(len(d_l[2])) if k not in pop_indices]
	print("Number of actual emails used in the `final`-ish dataset: "  + str(len(files_dl)))
	return pd.DataFrame({"Subject": subject_dl, "Body": body_dl, "File name": files_dl})

def checkReForward(subject):
	if ("re:" in subject) or ("fwd:" in subject) or ("fw:" in subject):
		return True

# DO this in the model building file instead - stopwording here does not work
def stopwords(line):
	for stop in stoplist:
		line = line.replace(stop, "")
	return line

def checkNonExtract(subject, txt):
	# print("sub" + subject)
	# print("txt" + txt)
	words = nltk.word_tokenize(subject)
	txt = nltk.word_tokenize(txt)
	# print(words)
	# print(txt)

	for word in words:
		if word not in txt:
			return False
	return True

def main():
	data_list = read_text()
	csv = create_csv(data_list)
	csv.to_csv("enron_cleaned.csv", sep = ',')
