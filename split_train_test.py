import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 读取CSV文件
data = pd.read_csv('keypoint.csv', header=None)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 划分训练和测试数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math

# 创建 DataLoader
train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes, num_nodes=21):
        super(TransformerModel, self).__init__()

        # Generate position encoding
        position = torch.arange(num_nodes).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.position_enc = torch.zeros(num_nodes, d_model)
        self.position_enc[:, 0::2] = torch.sin(position * div_term)
        self.position_enc[:, 1::2] = torch.cos(position * div_term)

        # Add learnable group embedding
        self.embedding = nn.Linear(2, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # Reshape input
        x = x.view(-1, 21, 2)  # x shape becomes [batch_size, 21, 2]

        # Embedding the input
        x = self.embedding(x)  # x shape becomes [batch_size, 21, d_model]

        # Add position encoding
        x += self.position_enc

        x = x.permute(1, 0, 2)  # Change shape to [seq_len, batch_size, d_model] as expected by Transformer
        x = self.encoder(x)
        x = self.classifier(x[0])  # Use the first token for classification
        return x
# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(42, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 14)  # 假设有10个类别
        )
    
    def forward(self, x):
        return self.fc(x)
    
    def load_model(self,path):
        return torch.load(path)
if __name__=='__main__':

    # model = SimpleModel()
    model=TransformerModel(d_model=24, nhead=4, num_layers=2, num_classes=14)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 在训练循环前初始化最小损失值
    min_loss = float('inf')  # 将其设置为无限大
    # 训练模型
    for epoch in range(100):  # 假设进行10个epoch的训练
        total_loss=0
        num=0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            total_loss=total_loss+loss.item()
            num+=1
            loss.backward()
            optimizer.step()
        # 检查并保存loss最低的模型
        if total_loss/num < min_loss:
            min_loss = total_loss/num
            torch.save(model, 'best_model.pth')
        print(f"Epoch {epoch+1}, Loss: {total_loss/num}")

    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    # 加载整个模型
    model_loaded = torch.load('best_model.pth')
    model=model_loaded
    model.eval()
    with torch.no_grad():
        test_data = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        y_true = []
        y_pred = []
        
        for batch_x, batch_y in test_loader:
            output = model(batch_x)
            _, predicted = torch.max(output.data, 1)
            
            y_true.extend(batch_y.tolist())
            y_pred.extend(predicted.tolist())

        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {accuracy}")

        # 可视化（混淆矩阵、准确率等）
        # 这里只简单地展示准确率
        plt.bar(["Accuracy"], [accuracy])
        plt.ylim([0, 1])
        plt.show()
