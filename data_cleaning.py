'''
data_cleaning.py

This file parses through the Enron email data text directory and returns a shortened csv file that
gets rid of all the reply/forward emails as well as emails with non-extractable words in the subjects.
It gets rid of symbols, numbers and unnecessary spaces, and emails with empty subject or body.
Thus, we can use the condensed email data for our next steps instead of running a huge chunk of data.
'''

import pandas as pd
import nltk
import os

# reads through all text files in the email file directory and includes all eligible ones
# in a nested list to further process into the final CSV
def read_text():
	data = [[],[],[]]

	for file in os.listdir('enron_all_emails/'):
		f = open('enron_all_emails/' + file, 'r', encoding="ascii", errors="surrogateescape")
		foundBody = False

		# check if email contains forward info in email body and gets rid of that
		foundBodyFwd = False
		foundBodySub = False
		# make sure that if body paragraph starts with forward/reply email but subject
		# does not indicate so, then that chunk of the paragraph is not ignored
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
		if (not body) or (not subject): # check for empty subject and body
			continue

		body.replace("=01&", '\'')
		body.replace("=01,", '\'')
		data[0].append(subject)
		data[1].append(body)
		data[2].append(file)

	return data

# the stopword list is quite cruicial in the process
stoplist = ['a', 'an', 'the', 'and', 'or', 'of', 'for', 'to', 'in', 'from', 'not', 'but', 'up']
symbols = '$`[]!@#$%^&*()_+-=;:",./<>?\\|~\{\}' # add dollar sign and ` and ''
escape = ['\n', '\t']

# filters all subjects and bodies and returns a Pandas dataframe
def create_csv(d_l):

	pop_indices = []

	for i in range(len(d_l[0])):
		subject = d_l[0][i].lower()
		line = d_l[1][i].lower()

		for ch in symbols:
			line = line.replace(ch, ' ')
			subject = subject.replace(ch, ' ')

		if checkReForward(subject):
			pop_indices.append(i)
			continue

		# replace apostrophes with empty string
		line = line.replace('\'', '')
		subject = subject.replace('\'', '')

		# remove numbers from string
		line = ''.join(m for m in line if not m.isdigit())
		subject = ''.join(n for n in subject if not n.isdigit())
        
        # use this instead of strip() to get rid of all empty spaces (including \n and \t)
		subject = ' '.join(subject.split())
		line = ' '.join(line.split())
        
        # check empty subject or body again
		if (not line) or (not subject):
			pop_indices.append(i)
			continue
            
		if not checkNonExtract(subject, line):
			pop_indices.append(i)
			continue

		if checkLength(line) > 500:
			pop_indices.append(i)
			continue

		# update data list
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

# checks to make sure that no reply/forward emails are included
def checkReForward(subject):
	if ("re " in subject) or ("fwd " in subject) or ("fw " in subject):
		return True
	return False

# returns the count of words in the body paragraph
def checkLength(body):
	words = nltk.word_tokenize(body)
	return len(words)

# returns True if all words in the subject line can be found in the body, otherwise False
def checkNonExtract(subject, txt):
	words = nltk.word_tokenize(subject)
	txt = nltk.word_tokenize(txt)

	for word in words:
		if word not in txt:
			return False
	return True

def main():
	data_list = read_text()
	csv = create_csv(data_list)
	csv.to_csv("enron_cleaned.csv", sep = ',')

main()
