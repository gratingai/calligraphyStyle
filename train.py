import data
import model
import evaluate
import torch
import torch.nn as nn
import torch.optim as optim



# 超参数
num_epochs = 10
device = 'mps'
batch_size = 16
lr = 0.002
momentum=0.9

best_accuracy = 0

# data
all_feature = data.LoadData(device=device)
test_data = all_feature.getData('test', batch_size=batch_size)
train_data = all_feature.getData('train', batch_size=batch_size)

# check data
for i, batch_item in enumerate(test_data):
    print(batch_item['pixel_values'].shape)
    print(batch_item['labels'])
    break

# model
model = model.create_model(len(all_feature.label2id), all_feature.label2id, all_feature.id2label, device=device)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)




# train running
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    index = 0
    for batch in train_data:
        if batch is None:  # 检查 batch 是否为 None
            continue  # 跳过 None 的批次
        
        optimizer.zero_grad()
        outputs = model(pixel_values=batch['pixel_values'],  labels=batch['labels'])  # 确保 pixel_values 传递为张量
        loss = criterion(outputs.logits, batch['labels'])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        index += 1

        if index % 200 == 0:
            accuracy, val_loss = evaluate.evaluate_model(model, test_data, criterion)
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{index}/{len(train_data)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}')
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), './workdir/best_model2.pth')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_data):.4f}')



