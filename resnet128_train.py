"""
ResNet128 深度学习训练脚本
使用 Weights & Biases (wandb) 进行实验跟踪、可视化和模型管理

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

# ============================================================================
# 1. 实验配置和wandb初始化
# ============================================================================

def init_wandb():
    """
    初始化wandb实验跟踪
    这里会创建一个新的实验运行，并设置项目名称
    """
    # 设置项目名称和实验名称
    project_name = "resnet128-training"
    experiment_name = f"resnet128-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # 初始化wandb
    wandb.init(
        project=project_name,
        name=experiment_name,
        config={
            # 模型超参数
            "model": "ResNet128",
            "num_classes": 10,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "Adam",
            "scheduler": "StepLR",
            "step_size": 30,
            "gamma": 0.1,
            
            # 数据增强参数
            "data_augmentation": True,
            "random_crop": 32,
            "random_horizontal_flip": 0.5,
            "normalize_mean": [0.4914, 0.4822, 0.4465],
            "normalize_std": [0.2023, 0.1994, 0.2010],
            
            # 训练参数
            "early_stopping_patience": 10,
            "save_best_model": True,
            "model_save_path": "./models"
        }
    )
    
    print(f"✅ Wandb实验已初始化: {experiment_name}")
    return wandb.config

# ============================================================================
# 2. ResNet128 模型定义
# ============================================================================

class BasicBlock(nn.Module):
    """
    ResNet的基本残差块
    包含两个3x3卷积层和跳跃连接
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # 主路径
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 跳跃连接
        out += self.shortcut(x)
        out = torch.relu(out)
        
        return out

class ResNet128(nn.Module):
    """
    ResNet128 模型
    包含多个残差层，用于图像分类任务
    """
    def __init__(self, num_classes=10):
        super(ResNet128, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 残差层组
        # 每组包含多个残差块，通道数逐渐增加
        self.layer1 = self._make_layer(64, 64, 2, stride=1)      # 64 -> 64
        self.layer2 = self._make_layer(64, 128, 2, stride=2)     # 64 -> 128
        self.layer3 = self._make_layer(128, 256, 2, stride=2)    # 128 -> 256
        self.layer4 = self._make_layer(256, 512, 2, stride=2)    # 256 -> 512
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        创建包含多个残差块的层
        """
        layers = []
        # 第一个块可能需要改变通道数和步长
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # 其余块保持相同的通道数和步长
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        初始化模型权重
        使用He初始化方法，适合ReLU激活函数
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始卷积
        x = torch.relu(self.bn1(self.conv1(x)))
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.fc(x)
        
        return x

# ============================================================================
# 3. 数据加载和预处理
# ============================================================================

def get_data_loaders(config):
    """
    创建训练和验证数据加载器
    包含数据增强和标准化
    """
    print("📊 正在准备数据加载器...")
    
    # 数据增强变换
    if config.data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(config.random_crop),
            transforms.RandomHorizontalFlip(config.random_horizontal_flip),
            transforms.ToTensor(),
            transforms.Normalize(config.normalize_mean, config.normalize_std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config.normalize_mean, config.normalize_std)
        ])
    
    # 验证集变换（无数据增强）
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.normalize_mean, config.normalize_std)
    ])
    
    # 加载CIFAR-10数据集
    # CIFAR-10包含10个类别的32x32彩色图像
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=val_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✅ 数据加载器创建完成")
    print(f"   - 训练集: {len(train_dataset)} 样本")
    print(f"   - 验证集: {len(val_dataset)} 样本")
    print(f"   - 批次大小: {config.batch_size}")
    
    return train_loader, val_loader

# ============================================================================
# 4. 训练和验证函数
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    训练一个epoch
    返回训练损失和准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [训练]')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    # 计算平均损失和准确率
    avg_loss = running_loss / len(train_loader)
    avg_acc = 100. * correct / total
    
    return avg_loss, avg_acc

def validate_epoch(model, val_loader, criterion, device):
    """
    验证一个epoch
    返回验证损失和准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度，节省内存
        pbar = tqdm(val_loader, desc='[验证]')
        
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 统计
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    # 计算平均损失和准确率
    avg_loss = running_loss / len(val_loader)
    avg_acc = 100. * correct / total
    
    return avg_loss, avg_acc

# ============================================================================
# 5. 主训练循环
# ============================================================================

def main():
    """
    主训练函数
    包含完整的训练流程和wandb集成
    """
    print("🚀 开始ResNet128训练流程...")
    
    # 1. 初始化wandb
    config = init_wandb()
    
    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    
    # 3. 创建模型
    print("🏗️ 正在创建ResNet128模型...")
    model = ResNet128(num_classes=config.num_classes).to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ 模型创建完成")
    print(f"   - 总参数数量: {total_params:,}")
    print(f"   - 可训练参数: {trainable_params:,}")
    
    # 4. 创建数据加载器
    train_loader, val_loader = get_data_loaders(config)
    
    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    
    # 6. 创建模型保存目录
    os.makedirs(config.model_save_path, exist_ok=True)
    
    # 7. 训练循环
    print(f"🎯 开始训练，共 {config.epochs} 个epoch...")
    
    best_val_acc = 0.0
    patience_counter = 0
    
    # 记录训练历史
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"{'='*60}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # 学习率调度
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 记录到wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr
        })
        
        # 保存训练历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 打印结果
        print(f"\n📊 Epoch {epoch+1} 结果:")
        print(f"   训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"   验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        print(f"   学习率: {current_lr:.6f}")
        
        # 检查是否为最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # 保存最佳模型
            if config.save_best_model:
                best_model_path = os.path.join(config.model_save_path, "best_model.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'config': dict(config)
                }, best_model_path)
                
                print(f"🏆 新的最佳模型已保存: {best_model_path}")
                
                # 将模型上传到wandb
                model_artifact = wandb.Artifact(
                    name=f"best-model-{wandb.run.id}",
                    type="model",
                    description="ResNet128最佳验证准确率模型"
                )
                model_artifact.add_file(best_model_path)
                wandb.log_artifact(model_artifact)
                print("✅ 最佳模型已上传到wandb")
        else:
            patience_counter += 1
            print(f"⏳ 验证准确率未提升，耐心计数: {patience_counter}/{config.early_stopping_patience}")
        
        # 早停检查
        if patience_counter >= config.early_stopping_patience:
            print(f"🛑 早停触发！验证准确率连续 {config.early_stopping_patience} 个epoch未提升")
            break
    
    # 8. 训练完成
    print(f"\n🎉 训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    # 保存最终模型
    final_model_path = os.path.join(config.model_save_path, "final_model.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_val_acc': val_acc,
        'best_val_acc': best_val_acc,
        'config': dict(config)
    }, final_model_path)
    
    print(f"📁 最终模型已保存: {final_model_path}")
    
    # 9. 创建训练曲线图
    create_training_plots(train_losses, train_accs, val_losses, val_accs)
    
    # 10. 完成wandb运行
    wandb.finish()
    print("✅ Wandb运行已完成")

def create_training_plots(train_losses, train_accs, val_losses, val_accs):
    """
    创建训练曲线图并上传到wandb
    """
    print("📈 正在创建训练曲线图...")
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='训练损失')
    ax1.plot(epochs, val_losses, 'r-', label='验证损失')
    ax1.set_title('训练和验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='训练准确率')
    ax2.plot(epochs, val_accs, 'r-', label='验证准确率')
    ax2.set_title('训练和验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率 (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = "training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 上传到wandb
    wandb.log({"training_curves": wandb.Image(plot_path)})
    print(f"✅ 训练曲线图已保存并上传到wandb: {plot_path}")

# ============================================================================
# 6. 脚本入口
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        wandb.finish()
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {str(e)}")
        wandb.finish()
        raise e 