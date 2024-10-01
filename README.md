## Project Overview

This project involves building a sequence-to-sequence (Seq2Seq) neural machine translation model using an encoder-decoder architecture with attention mechanisms. The model is implemented in PyTorch and trained on English-to-German translation tasks. Once the model is trained, it can translate German sentences into English. The project emphasizes efficiency, allowing the model to be saved and reloaded without retraining, which significantly reduces computation time during the evaluation phase.

### Key Components

	1.	EncoderRNN: A Recurrent Neural Network (RNN) encoder that processes the input sentence and generates context vectors (hidden states).
	2.	AttnDecoderRNN: An attention-based decoder that produces the translated sentence, focusing on relevant parts of the input sequence at each decoding step.
	3.	Training & Evaluation: The model is trained using batches of English-German sentence pairs, and once trained, the encoder and decoder models are saved for future evaluation without retraining.

### Project Files

	•	lang_translate.ipynb: The main Jupyter Notebook containing the full project code, including model definitions, training, evaluation, and saving/loading mechanisms.
	•	encoder_model.pth & decoder_model.pth: Saved models for the encoder and decoder, allowing future evaluations without retraining.


 # Seq2Seq Language Translation with Attention

This project implements a neural machine translation model using an encoder-decoder architecture with attention. The model is trained on English-German sentence pairs and can translate German sentences into English. The project is built using PyTorch and takes advantage of attention mechanisms to improve translation quality.

## Features
- **Encoder-Decoder Architecture**: Processes input and generates output sequences.
- **Attention Mechanism**: Focuses on specific parts of the input when generating each word in the output.
- **Model Persistence**: The trained model can be saved and reloaded, allowing users to skip the time-consuming training phase and directly evaluate the model on new input sentences.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Jupyter Notebook

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/K-Ashik/-Neural-Machine-Translation-with-Attention-Using-Seq2Seq-Architecture.git
   cd seq2seq-translation


## How to Use

	1.	Training the Model: Run the code in lang_translate.ipynb to train the model. However, the pre-trained models are provided, so this step can be skipped if not needed.
	2.	Saving and Loading the Model: After training, the encoder and decoder models are saved as encoder_model.pth and decoder_model.pth.
	3.	Evaluating the Model: You can evaluate new sentences using the pre-trained model without rerunning the entire training process.


## Load the saved models
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

encoder.load_state_dict(torch.load('encoder_model.pth', map_location=device))
decoder.load_state_dict(torch.load('decoder_model.pth', map_location=device))

encoder.eval()
decoder.eval()

# Evaluate and translate a sentence
evaluateAndShowAttention('er ist ein begabter schachspieler')


# Model Architecture

	•	EncoderRNN: A GRU-based encoder that processes the input sentence and generates hidden states.
	•	AttnDecoderRNN: An attention-based decoder that generates the translated sentence using the context vectors from the encoder and the attention mechanism.

# Example Translations

	•	Input: er ist ein begabter schachspieler
	•	Output: he is a talented chess player
	•	Input: ich bin das schonste einhorn der welt
	•	Output: i am the most beautiful unicorn in the world

