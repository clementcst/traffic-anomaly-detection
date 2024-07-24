import torch
import torch.nn as nn
import torch.optim as optim
from SKTD_utils import measure_performance
import cProfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiScaleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels - 2*(out_channels // 3), kernel_size=5, padding=2)
    
    def forward(self, x):
        y1 = self.conv1(x)
        y3 = self.conv3(x)
        y5 = self.conv5(x)
        return torch.cat([y1, y3, y5], dim=1)

class MSSTRBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MSSTRBlock, self).__init__()
        self.multi_scale_conv = MultiScaleConv1D(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.residual_conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)
    
    def forward(self, x):
        identity = x
        x = x.transpose(1, 2)  # (batch, time, features) -> (batch, features, time)
        x = self.multi_scale_conv(x)
        identity = self.residual_conv(identity.transpose(1, 2)).transpose(1, 2)
        x = x.transpose(1, 2)  # (batch, features, time) -> (batch, time, features)
        x, _ = self.lstm(x)
        x = self.norm(x)
        x = self.relu(x + identity)  # Residual connection
        return x

class MSSTRNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_blocks):
        super(MSSTRNet, self).__init__()
        self.initial_conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.blocks = nn.ModuleList([MSSTRBlock(hidden_size, hidden_size) for _ in range(num_blocks)])
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, time, features) -> (batch, features, time)
        x = self.initial_conv(x)
        x = x.transpose(1, 2)  # (batch, features, time) -> (batch, time, features)
        
        for block in self.blocks:
            x = block(x)
        
        # Global Average Pooling
        x = x.mean(dim=1)
        
        return self.fc(x)

class LENet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LENet, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size // 2, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_size // 2, hidden_size // 2, batch_first=True)
        self.fc = nn.Linear(hidden_size // 2, num_classes)
    
    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))

# Fonction de perte focale
def focal_loss(predictions, targets, alpha=0.25, gamma=2):
    ce_loss = nn.CrossEntropyLoss(reduction='none')(predictions, targets)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    return focal_loss.mean()

# entainement modele Teacher
@measure_performance
def train_msstrnet(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100*correct/total:.2f}%')

    return model

# entainement modele student
@measure_performance
def train_lenet_kd(teacher_model, student_model, train_loader, val_loader, epochs=50, lr=0.001, temperature=3.0, lambda_kd=0.5, device='cpu'):
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        student_model.train()
        teacher_model.eval()
        
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_logits = teacher_model(batch_x)
            
            student_logits = student_model(batch_x)
            
            soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=1)
            soft_prob = nn.functional.log_softmax(student_logits / temperature, dim=1)

            soft_loss = nn.KLDivLoss(reduction='batchmean')(soft_prob, soft_targets) * (temperature**2)
            hard_loss = focal_loss(student_logits, batch_y)
            
            loss = lambda_kd * soft_loss + (1 - lambda_kd) * hard_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        student_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = student_model(batch_x)
                val_loss += focal_loss(outputs, batch_y).item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100*correct/total:.2f}%')
    
    return student_model