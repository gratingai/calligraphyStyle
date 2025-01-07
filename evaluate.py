import torch


def evaluate_model(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            data, label = batch['pixel_values'], batch['labels']
                
            outputs = model(pixel_values=data)
            loss = criterion(outputs.logits, label)
            total_loss += loss.item()
            
            predictions = outputs.logits.argmax(-1)
            correct += (predictions == label).sum().item()
            total += len(predictions)
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0
    return accuracy, avg_loss
