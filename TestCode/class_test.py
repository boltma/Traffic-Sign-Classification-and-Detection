## Usage: python test.py --predfile pred.json --labelfile test.json

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--predfile', type=str, default='pred.json')
parser.add_argument('--labelfile', type=str, default='test.json')
args = parser.parse_args()

pred = json.load(open(args.predfile, 'r'))
label = json.load(open(args.labelfile, 'r'))

classes = []
correct = {}
total = {}
for cls in label.values():
	if cls not in classes:
		classes.append(cls)
		correct[cls] = 0
		total[cls] = 0
classes.sort()

miss = 0
cor = 0
for imgname in label.keys():
	try:
		correct[label[imgname]] += (pred[imgname] == label[imgname])
	except:
		miss += 1
	total[label[imgname]] += 1
acc_str = '%d imgs missed\n'%miss
for cls in classes:
	acc_str += 'class:%s\trecall:%f\n'%(cls, correct[cls]/total[cls])
	cor += correct[cls]
acc_str += 'Accuracy: %f'%(cor/len(label))
print(acc_str)