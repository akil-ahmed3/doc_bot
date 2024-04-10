# GPT-2 Text Generation with Custom Dataset

This repository contains code for training and using the GPT-2 model from Hugging Face with a custom dataset. The model is fine-tuned on the custom dataset to create a chatbot based on the conversations between real Doctors and patients.

## Dataset

The dataset used for training the GPT-2 model can be found at the following URLs:

1. [IClinic Dataset](https://github.com/LasseRegin/medical-question-answer-data.git)
   

2. [MedQuad Dataset](https://github.com/abachaa/MedQuAD)

Description: These dataset contains conversations between real Doctors and patients.

Please download the dataset(s) and preprocess them as according to preprocessing.ipynb before training the model.

## Model

We utilize the GPT-2 model from Hugging Face's Transformers library. The model is pre-trained on a large corpus of text data and fine-tuned on the custom dataset for text generation tasks.

### Model Information

- Model: GPT-2
- Model Size: [Insert model size, e.g., "Small", "Medium", "Large", "XL"]
- Hugging Face Model Hub: [Link to model on Hugging Face Model Hub](Model_Hub_Link)
- PyTorch Model ID: [Insert PyTorch model ID, if applicable]


### Training

To train the GPT-2 model on the custom dataset, follow these steps:

1. Download the dataset(s) from the provided URLs.
2. Preprocess steps are in preprocessing.ipynb file.
3. Fine-tune the GPT-2 model on the dataset using the provided training script.

### Inference

Once the model is trained, you can use it to answer medical questions that you might have, by providing prompts or input sequences. Here's how to use the trained model for inference:

1. Load the fine-tuned model checkpoint.
2. Provide input text or prompts to generate text using the model.

### Example Code

```python
# Example code for text generation with the fine-tuned GPT-2 model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned GPT-2 model checkpoint
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Generate text
input_text = "what is appendix?"
input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)

output = model.generate(
                                input_ids,
                                do_sample=True,
                                top_k=50,
                                max_length = 100,
                                top_p=0.95,
                                num_return_sequences=3
                                )

# Decode and print generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
