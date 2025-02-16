from abc import ABC
import datasets
import torch
import random
import json
import time
from torch.utils.data import Subset, DataLoader, Dataset, RandomSampler, SequentialSampler
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer
from nltk.wsd import lesk
from tqdm import tqdm
def get_synsets_for_word(word):
	synsets = wn.synsets(word)
	return synsets
def get_word_token_indices(tokens, word, tokenizer):
	word_tokens = tokenizer.tokenize(word)
	word_len = len(word_tokens)
	indices = []
	for i in range(len(tokens) - word_len + 1):
		if tokens[i:i + word_len] == word_tokens:
			indices.append(list(range(i, i + word_len)))
	return indices

def get_train_val_sstdata_init(stage = 0, batch_size=64):
	dataset_val = SSTdata(subset='val')
	#tokenizer = dataset_train.get_tokenizer()
	#vocab_size = tokenizer.vocab_size
	val_dataloader = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)
	return val_dataloader
def get_train_val_loaders_sstdata(stage = 1, batch_size=64):
	dataset_train = SSTdata(subset='train',  stage = stage)
	dataset_val = SSTdata(subset='val')
	#tokenizer = dataset_train.get_tokenizer()
	#vocab_size = tokenizer.vocab_size
	train_dataloader = DataLoader(dataset_train, batch_size=batch_size)
	val_dataloader = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)
	return train_dataloader, val_dataloader
class SSTdata(Dataset, ABC):
	def __init__(self,  subset='train', max_length=100,  stage = None):
		if subset=='train':
			file_path = r"sorted_sst2.json"
			sentences, self.labels = [], []
			with open(file_path, "r", encoding="utf-8") as f:
				for line in f:  # Read line by line
					data = json.loads(line.strip())  # Parse each line as JSON
					sentences.append(data["sentence"])
					self.labels.append(data["label"])
			if stage == 1:
				sentences = sentences[:int(len(sentences) * 0.6)]
				self.labels = self.labels[:int(len(self.labels) * 0.6)]
			elif stage == 2:
				sentences = sentences[:int(len(sentences) * 0.8)]
				self.labels = self.labels[:int(len(self.labels) * 0.8)]
			elif stage == 3:
				sentences = sentences[:int(len(sentences) * 1)]
				self.labels = self.labels[:int(len(self.labels) * 1)]
		elif subset =='val':
			self.data = datasets.load_dataset("glue", "sst2")['validation']
			sentences = self.data['sentence']
			self.labels = self.data['label']
		else:
			self.data = datasets.load_dataset("glue", "sst2")['test']
		if subset == 'train':
			data = [{"sentence": s, "label": l} for s, l in zip(sentences, self.labels)]
			oversampled_data = self._oversample(data)

			# Now extract the sentences and labels after oversampling
			sentences = [item["sentence"] for item in oversampled_data]
			self.labels = [item["label"] for item in oversampled_data]
		self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
		#self.data = pd.read_json(self.path,lines=True, orient='records')
		
		encodings = []
		for sentence in sentences:
			encoded_data = self.tokenizer.encode_plus(sentence, max_length=max_length, padding='max_length'
                                                 , truncation=True)
			encodings.append(encoded_data)
		self.encodings = encodings
	def _oversample(self, data):

		type_of_each = {}

		maxim = 0
		for item in data:
			label = item["label"]
			if label not in type_of_each:
				type_of_each[label] = 1
			else:
				type_of_each[label] += 1
			if type_of_each[label] > maxim:
				maxim = type_of_each[label]

		oversample_factor = {}
		for key in type_of_each:
			oversample_factor[key] = maxim // type_of_each[key]

		samples = []
		for item in data:
			label = item["label"]
			samples += [item] * oversample_factor[label]

		random.shuffle(samples)

		print(type_of_each)
        
		type_of_each = {}
		for item in samples:
			label = item["label"]
			if label not in type_of_each:
				type_of_each[label] = 1
			else:
				type_of_each[label] += 1
                
		print(type_of_each)
        
		return samples
	def __getitem__(self, idx):
		item = {}
		temp_input_ids = self.encodings[idx]['input_ids']
		#masked_input_ids= self.mask_tokens(temp_input_ids)
		item['input_ids'] = torch.tensor(temp_input_ids)
		item['attention_masks'] = torch.tensor(self.encodings[idx]['attention_mask'])
		item['token_type_ids'] = torch.tensor(self.encodings[idx]['token_type_ids'])
		item['labels'] = torch.tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.encodings)
	'''
	def mask_tokens(self, input_ids):
		
		masked_input_ids = input_ids[:]
		for i in range(len(input_ids)):
		    if random.random() < self.mask_prob:
		        masked_input_ids[i] = self.tokenizer.mask_token_id 
		return masked_input_ids
		
		tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
		words = self.tokenizer.convert_tokens_to_string(tokens).split()
		
		word_indices = {}
		for word in self.sentiment_dict:
			indices = get_word_token_indices(tokens, word, self.tokenizer)
			if indices:
				word_indices[word] = indices
		
		words_to_mask = set(word for word in tokens if word in self.sentiment_dict)
		word_indices = {}
		for word in words_to_mask:
			word_indices1 = get_word_token_indices(tokens, word, self.tokenizer)
			word_indices[word] = word_indices1
		sorted_words = sorted(word_indices.keys(), key=lambda x: abs(self.sentiment_dict[x]), reverse=True)
		total_words = len(words)
		num_to_mask = max(1, int(total_words * self.mask_prob))

		masked_tokens = tokens.copy()
		masked_word_count = 0
		for word in sorted_words:
			if masked_word_count >= num_to_mask:
				break 
			for indices in word_indices[word]:
				for idx in indices:
					if masked_tokens[idx] not in self.tokenizer.all_special_tokens:  # Exclude special tokens
						masked_tokens[idx] = '[MASK]'
			masked_word_count += 1
		return self.tokenizer.convert_tokens_to_ids(masked_tokens)
	'''
	def get_tokenizer(self):
		return self.tokenizer
	def mask_tokens(self, input_ids):
		tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
		words = self.tokenizer.convert_tokens_to_string(tokens).split()

		#words_to_mask = set(word for word in words if word in self.sentiment_dict)
		'''
		synsets = {}
		y = time.time()
		for word in words:
			context = words
			synset = lesk(context, word)
			if synset:
				pos_id = f"{synset.pos()}.{synset.offset():08d}"
				synsets[word] = pos_id
			else:
				synsets[word] = None
		print(synsets,time.time()-y)
		exit()
		'''
		'''
		y = time.time()
		words_to_mask = set()
		for word in words:
			l = get_synsets_for_word(word)
			for k in l:
				k = f"{k.pos()}.{k.offset():08d}"
				if k in self.sentiment_dict:
					words_to_mask.add(word)
					break
		#print(words_to_mask)
		'''
		word_indices = {}
		word_objectivity_scores = {}
		#print(time.time()-y)
		#y = time.time()
		'''
		for word in words:
			synsets = get_synsets_for_word(word)  # You need to implement this function
			best_score = None
			best_synset = None

			for synset in synsets:
				synset = f"{synset.pos()}.{synset.offset():08d}"
				if synset in self.sentiment_dict:
					pos_score = self.sentiment_dict[synset]['pos_score']
					neg_score = self.sentiment_dict[synset]['neg_score']
					objectivity_score = 1 - (pos_score + neg_score)

					if best_score is None or objectivity_score < best_score:
						best_score = objectivity_score
						best_synset = synset

			if best_synset:
				word_objectivity_scores[word] = best_score
				word_indices[word] = get_word_token_indices(tokens, word, self.tokenizer)
		'''
		words = [te for te in words if te not in ['[CLS]','[SEP]','[PAD]']]
		for word in words:
			context = words
			synset = lesk(context, word)
			if synset:
				pos_id = f"{synset.pos()}.{synset.offset():08d}"
				if pos_id in self.sentiment_dict:
					pos_score = self.sentiment_dict[pos_id]['pos_score']
					neg_score = self.sentiment_dict[pos_id]['neg_score']
					objectivity_score = 1 - (pos_score + neg_score)
					word_objectivity_scores[word] = objectivity_score
					word_indices[word] = get_word_token_indices(tokens, word, self.tokenizer)
				else:
					word_objectivity_scores[word] = 1.
					word_indices[word] = get_word_token_indices(tokens, word, self.tokenizer)
			else:
				word_objectivity_scores[word] = 1.
				word_indices[word] = get_word_token_indices(tokens, word, self.tokenizer)
		#print(time.time()-y)
		#exit()
	    # Sort words by their objectivity score (lower score means less objective)

		sorted_words = sorted(word_objectivity_scores.keys(), key=lambda x: word_objectivity_scores[x])

		tokens_temp = [te for te in tokens if te not in ['[CLS]','[SEP]','[PAD]']]
		total_words = len(tokens_temp)

		num_to_mask = max(1, int(total_words * self.mask_prob*3)) # *3
		masked_tokens = tokens.copy()
		masked_word_count = 0

		'''
		for word in sorted_words:
			if masked_word_count >= num_to_mask:
				break 
			for indices in word_indices[word]:
				for idx in indices:
					if masked_tokens[idx] not in self.tokenizer.all_special_tokens:  # Exclude special tokens
						masked_tokens[idx] = '[MASK]'
			masked_word_count += 1
		'''
		#(1 + word_objectivity_scores[word]) * 
		#(2 - word_objectivity_scores[word]) * 

		cnter = 0
		for word in sorted_words:
			for indices in word_indices[word]:
				for idx in indices:
					if cnter > num_to_mask:
						return self.tokenizer.convert_tokens_to_ids(masked_tokens)
					if random.random() <=  (2 - word_objectivity_scores[word]) * self.mask_prob:
						if masked_tokens[idx] not in self.tokenizer.all_special_tokens:  # Exclude special tokens
							masked_tokens[idx] = '[MASK]'
							cnter += 1
		return self.tokenizer.convert_tokens_to_ids(masked_tokens)

	def update_mask_prob(self, new_mask_prob):
		self.mask_prob = new_mask_prob