'''
	Execute locally with: 
		convert_vocab_for_hf_tokenizer.py <in_vocab_filepath> <out_vocab_filepath>
'''

import sys

def parse_vocab(in_vocab_filepath):
	with open(in_vocab_filepath, "r") as in_vocab_file:
		original_vocab = in_vocab_file.read()

	lines = original_vocab.split("\n")
	raw_tokens = [x.strip() for x in lines] # remove spaces in start of line
	raw_tokens = [x[:-1] if x[-1] == "," else x for x in raw_tokens] ## remove "," between lines

	#print(raw_tokens)
	tokens = []
	for t in raw_tokens:
		if t[0] == "\"" and t[-1] == "\"":
			tokens.append(t[1:-1])
		else:
			print("token {} invalid: t[0] = {}, t[-1] = {}".format(t, t[0], t[-1]))

	#print(tokens)

	word_start_tokens = []
	in_word_tokens = []
	for t in tokens:
		if "\\u2581" in t:
			word_start_tokens.append(t.split("\\u2581")[1])
		else:
			in_word_tokens.append("##" + t)

	#print(word_start_tokens)
	#print(in_word_tokens)

	## lowercase - so we use bert-base-uncase
	converted_vocab = word_start_tokens + in_word_tokens
	converted_vocab = [t.lower() for t in converted_vocab]
	return converted_vocab

def main():
	if len(sys.argv) != 3:
		print("USAGE: convert_vocab_for_hf_tokenizer.py <in_vocab_filepath> <out_vocab_filepath>")
		exit(-1)

	in_vocab_filepath = sys.argv[1]
	out_vocab_filepath = sys.argv[2]

	tokenizer_vocab = parse_vocab(in_vocab_filepath)

	## remove special tokens
	tokenizer_vocab.remove("##<s>")
	tokenizer_vocab.remove("##</s>")
	tokenizer_vocab.remove("##<unk>")

	with open(out_vocab_filepath, "w+") as out_vocab_file:
		out_vocab_file.write("\n".join(tokenizer_vocab))

if __name__ == "__main__":
	main()
