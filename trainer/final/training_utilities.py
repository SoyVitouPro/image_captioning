import jiwer
import re
import torch
import json
from config.config import idx2word_path


def decode_indices(indices, idx2word):
    characters = []
    for index in indices:
        character = idx2word.get(str(index))
        if character is None:
            return None, f"Index '{index}' not found in idx2word!"
        characters.append(character)
    return ''.join(characters), None

# Function to load idx2word and convert it to word2idx
def load_vocabulary(path):
    with open(path, 'r') as file:
        idx2word = json.load(file)
    word2idx = {v: int(k) for k, v in idx2word.items()}
    return idx2word, word2idx

# Load vocabulary
idx2word, word2idx = load_vocabulary(idx2word_path)

def custom_transform(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Return as list of words
    return text.split()
    

def calculate_wer(gt, pred, epoch, file_path='metric.txt'):
    
    with open(file_path, 'a') as file:  # Open file in append mode
        file.write(f"Epoch {epoch}\n")
        
        file.write("===========================\n")
        match_pred = re.search(r"^(.*?)<END>", pred)
        if match_pred:
            content_pred = match_pred.group(1)
        file.write(f"pred: {content_pred}\n")
        match_ground_true = re.search(r"<START>(.*?)<END>", gt)
        if match_ground_true:
            content_ground_true = match_ground_true.group(1)
        file.write(f"true: {content_ground_true}\n")
        file.write("===========================\n")
    
    if not content_ground_true:  # Ensure non-empty
        content_ground_true = ['']
    if not content_pred:  # Ensure non-empty
        content_pred = ['']
    wer_score = jiwer.wer(content_ground_true, content_pred)
    
    return wer_score

def calculate_cer(gt, pred):
    match_pred = re.search(r"^(.*?)<END>", pred)
    if match_pred:
        content_pred = match_pred.group(1)
    
    match_ground_true = re.search(r"<START>(.*?)<END>", gt)
    if match_ground_true:
        content_ground_true = match_ground_true.group(1)

    return jiwer.cer(content_ground_true, content_pred)

def evaluate_model(encoder, decoder, dataloader, device, epoch):
    encoder.eval()
    decoder.eval()
    total_wer, total_cer, num_samples = 0, 0, 0
    with torch.no_grad():
        for images, captions in dataloader:
            images, captions = images.to(device), captions.to(device)
            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])
            predicted_captions = outputs.argmax(-1)
            
            for i in range(len(captions)):
                gt_caption = decode_indices(captions[i].tolist(), idx2word)[0]
                pred_caption = decode_indices(predicted_captions[i].tolist(), idx2word)[0]
                
                wer = calculate_wer(gt_caption, pred_caption, epoch)
                cer = calculate_cer(gt_caption, pred_caption)
                total_wer += wer
                total_cer += cer
                num_samples += 1

    avg_wer = total_wer / num_samples if num_samples > 0 else 0
    avg_cer = total_cer / num_samples if num_samples > 0 else 0
    print(f"Average WER: {avg_wer:.2f}, Average CER: {avg_cer:.2f}")
    return avg_wer, avg_cer