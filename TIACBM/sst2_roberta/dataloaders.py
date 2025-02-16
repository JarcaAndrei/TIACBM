from abc import ABC
import datasets
import torch
import random
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
def get_train_val_loaders_sstdata(probs, epoch, sentiment_dict, batch_size=64):
	dataset_train = SSTdata(subset='train', mask_prob=probs[epoch], sentiment_dict = sentiment_dict)
	dataset_val = SSTdata(subset='val', mask_prob = 0)
	#tokenizer = dataset_train.get_tokenizer()
	#vocab_size = tokenizer.vocab_size
	train_dataloader = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
	val_dataloader = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)
	return train_dataloader, val_dataloader
class SSTdata(Dataset, ABC):
	def __init__(self, sentiment_dict = None, subset='train', max_length=100, mask_prob = 0.99):
		if subset=='train':
			self.data = datasets.load_dataset("glue", "sst2")['train']
		elif subset =='val':
			self.data = datasets.load_dataset("glue", "sst2")['validation']
		else:
			self.data = datasets.load_dataset("glue", "sst2")['test']
		self.mask_prob = mask_prob
		self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", do_lower_case=True)
		self.sentiment_dict = sentiment_dict
		sentences = self.data['sentence']
		self.labels = self.data['label']
		encodings = []
		for sentence in sentences:
			encoded_data = self.tokenizer.encode_plus(sentence, max_length=max_length, padding='max_length'
                                                 , truncation=True)
			encodings.append(encoded_data)
		self.encodings = encodings
		if subset != 'val' and self.mask_prob != 0:
			for i in tqdm(range(len(self.encodings)), desc="Masking tokens"):
				self.encodings[i]['input_ids'] = self.mask_tokens(self.encodings[i]['input_ids'])
	def __getitem__(self, idx):
		item = {}
		temp_input_ids = self.encodings[idx]['input_ids']
		item['input_ids'] = torch.tensor(temp_input_ids)
		item['attention_masks'] = torch.tensor(self.encodings[idx]['attention_mask'])
		item['labels'] = torch.tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.encodings)
	
	def get_tokenizer(self):
		return self.tokenizer
	def mask_tokens(self, input_ids):
		tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
		words = self.tokenizer.convert_tokens_to_string(tokens).split()

		word_indices = {}
		word_objectivity_scores = {}
		
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
		
		sorted_words = sorted(word_objectivity_scores.keys(), key=lambda x: word_objectivity_scores[x])

		tokens_temp = [te for te in tokens if te not in ['[CLS]','[SEP]','[PAD]']]
		total_words = len(tokens_temp)

		num_to_mask = max(1, int(total_words * self.mask_prob*3)) # *3
		masked_tokens = tokens.copy()
		masked_word_count = 0

		cnter = 0
		for word in sorted_words:
			for indices in word_indices[word]:
				for idx in indices:
					if cnter > num_to_mask:
						return self.tokenizer.convert_tokens_to_ids(masked_tokens)
					if random.random() <= (2 - word_objectivity_scores[word]) * self.mask_prob:
						if masked_tokens[idx] not in self.tokenizer.all_special_tokens: 
							masked_tokens[idx] = '[MASK]'
							cnter += 1
		return self.tokenizer.convert_tokens_to_ids(masked_tokens)

	def update_mask_prob(self, new_mask_prob):
		self.mask_prob = new_mask_prob