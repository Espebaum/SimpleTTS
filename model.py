import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTTS(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, mel_dim):
		super(SimpleTTS, self).__init__()
		# Text Encoder
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.encoder_rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

		# Mel Decoder
		self.decoder_rnn = nn.GRU(mel_dim, hidden_dim, batch_first=True)
		self.mel_linear = nn.Linear(hidden_dim, mel_dim) # (256, 100)

		# Stop Token Predictor
		self.stop_token = nn.Linear(hidden_dim, 1)

	def	forward(self, text, mel_input):
		"""
		Encode Text : 
			-> text.shape ([2, 50]) 
		Decode Mel Spectrogram : 
			-> mel_input.shape ([2, 100, 80])
		"""

		# Encode text ([2, 50])
		text_embedded = self.embedding(text) # (batch, text_len, embedding_dim) 
		# ([2, 50, 128])
		encoder_outputs, _ = self.encoder_rnn(text_embedded) # (batch, text_len, hidden_dim)
		# ([2, 50, 256])

		# Decode mel Spectrogram ([2, 100, 80])
		decoder_outputs, _ = self.decoder_rnn(mel_input) # (batch, mel_len, embedding_dim)
		# ([2, 100, 256])

		mel_output = self.mel_linear(decoder_outputs) # ([2, 100, 80])

		# Stop token prediction
		stop_output = self.stop_token(decoder_outputs) # ([2, 100, 1])
		# print(stop_output.shape) 
		stop_output = stop_output.squeeze(-1) # ([2, 100])
		
		return mel_output, stop_output
