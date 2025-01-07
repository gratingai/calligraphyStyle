import data
import model
import evaluate
import torch
import torch.nn as nn
import torch.optim as optim

device = 'mps'
batch_size = 4

# 数据加载
all_feature = data.InitData(device=device)
test_loader = all_feature.getCheckData(batch_size=batch_size)

print('debug len:test_data', len(test_loader))
for i, batch_item in enumerate(test_loader):
    print(batch_item['pixel_values'].shape)
    print(batch_item['labels'])
    break

# 模型初始化
model = model.create_model(len(all_feature.label2id), all_feature.label2id, all_feature.id2label, device=device)

# 加载预训练模型
model_path = './workdir/best_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # 设置模型为评估模式

# 进行预测
with torch.no_grad():  # 禁用梯度计算，节省内存
    for i, batch in enumerate(test_loader):
        pixel_values = batch['pixel_values']
        labels = batch['labels']

        outputs = model(pixel_values=pixel_values)
        _, predicted = torch.max(outputs.logits, 1)  # 获取预测的类别

        # 打印预测结果
        if i%200 != 0: continue
        for j in range(len(predicted)):
            print(f'  Sample {j+1}: True Label: {all_feature.id2label[labels[j].item()]}  Predicted Label: {all_feature.id2label[predicted[j].item()]}')
