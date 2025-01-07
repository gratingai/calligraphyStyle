import argparse  # 添加argparse库
import model
import torch
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
import os

device = 'cuda'
label2id = {'lqs': 0, 'htj': 1, 'wzm': 2, 'fwq': 3, 'yzq': 4, 'hy': 5, 'mf': 6, 'sgt': 7, 'smh': 8, 'oyx': 9, 'lx': 10, 'wxz': 11, 'zmf': 12, 'yyr': 13, 'mzd': 14, 'bdsr': 15, 'csl': 16, 'gj': 17, 'shz': 18, 'lgq': 19}
id2label = {0: 'lqs', 1: 'htj', 2: 'wzm', 3: 'fwq', 4: 'yzq', 5: 'hy', 6: 'mf', 7: 'sgt', 8: 'smh', 9: 'oyx', 10: 'lx', 11: 'wxz', 12: 'zmf', 13: 'yyr', 14: 'mzd', 15: 'bdsr', 16: 'csl', 17: 'gj', 18: 'shz', 19: 'lgq'}
        
# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Calligraphy Prediction API')
parser.add_argument('--model-path', type=str, default='./workdir/best_model.pth', help='Path to the model file')
args = parser.parse_args()

# 模型初始化
model = model.create_model(len(label2id), label2id, id2label, device=device)

# 加载预训练模型
model_path = args.model_path
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # 设置模型为评估模式

# 定义图片预处理函数
def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0).to(device)

# 定义预测函数
def predict_image(image_bytes):
    with torch.no_grad():
        pixel_values = preprocess_image(image_bytes)
        outputs = model(pixel_values=pixel_values)
        _, predicted = torch.max(outputs.logits, 1)
        return label_to_calligrapher_name(id2label[predicted.item()])


def label_to_calligrapher_name(label):
    # 创建一个字典来存储Label到Calligrapher Name的映射
    label_to_name = {
        'wxz': '王羲之',
        'yzq': '颜真卿',
        'lgq': '柳公权',
        'sgt': '孙过庭',
        'smh': '沙孟海',
        'mf': '米芾',
        'htj': '黄庭坚',
        'oyx': '欧阳询',
        'zmf': '赵孟頫',
        'csl': '褚遂良',
        'wzm': '文征明',
        'lqs': '梁秋生',
        'yyr': '于右任',
        'hy': '弘一',
        'bdsr': '八大山人',
        'fwq': '范文强',
        'gj': '管峻',
        'shz': '宋徽宗',
        'mzd': '毛泽东',
        'lx': '鲁迅'
    }
    
    # 返回对应的Calligrapher Name，如果没有找到则返回None
    return label_to_name.get(label, None)


# 创建Flask应用
app = Flask(__name__)

# 定义API接口
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    image_file = request.files['image']
    image_bytes = image_file.read()
    prediction = predict_image(image_bytes)
    return jsonify({'prediction': prediction})

# 启动Flask应用
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)