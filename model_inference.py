"""
ResNet128 模型推理脚本
用于加载训练好的模型并进行预测

"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from resnet128_train import ResNet128

# CIFAR-10类别名称
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_trained_model(model_path, num_classes=10, device='cpu'):
    """
    加载训练好的模型
    
    参数:
        model_path: 模型文件路径
        num_classes: 类别数量
        device: 计算设备
    
    返回:
        加载了权重的模型
    """
    print(f"📥 正在加载模型: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 创建模型实例
    model = ResNet128(num_classes=num_classes).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式
    model.eval()
    
    print(f"✅ 模型加载成功")
    print(f"   - 训练轮次: {checkpoint.get('epoch', 'N/A')}")
    print(f"   - 最佳验证准确率: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    
    return model, checkpoint

def preprocess_image(image_path, target_size=32):
    """
    预处理图像用于模型输入
    
    参数:
        image_path: 图像文件路径
        target_size: 目标尺寸
    
    返回:
        预处理后的图像张量
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 调整图像大小
    image = image.resize((target_size, target_size))
    
    # 转换为张量并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    # 添加批次维度
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image

def predict_single_image(model, image_tensor, device):
    """
    对单张图像进行预测
    
    参数:
        model: 训练好的模型
        image_tensor: 预处理后的图像张量
        device: 计算设备
    
    返回:
        预测结果和概率
    """
    with torch.no_grad():
        # 将图像移到设备上
        image_tensor = image_tensor.to(device)
        
        # 前向传播
        outputs = model(image_tensor)
        
        # 获取预测类别和概率
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # 获取所有类别的概率
        all_probabilities = probabilities[0].cpu().numpy()
        
        return predicted_class, confidence, all_probabilities

def visualize_prediction(image, predicted_class, confidence, all_probabilities, save_path=None):
    """
    可视化预测结果
    
    参数:
        image: 原始图像
        predicted_class: 预测的类别
        confidence: 预测置信度
        all_probabilities: 所有类别的概率
        save_path: 保存路径
    """
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 显示图像
    ax1.imshow(image)
    ax1.set_title(f'预测结果: {CIFAR10_CLASSES[predicted_class]} (置信度: {confidence:.2%})')
    ax1.axis('off')
    
    # 显示概率分布
    bars = ax2.bar(range(len(CIFAR10_CLASSES)), all_probabilities)
    ax2.set_xlabel('类别')
    ax2.set_ylabel('概率')
    ax2.set_title('各类别预测概率')
    ax2.set_xticks(range(len(CIFAR10_CLASSES)))
    ax2.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
    
    # 高亮预测的类别
    bars[predicted_class].set_color('red')
    
    # 添加概率值标签
    for i, prob in enumerate(all_probabilities):
        ax2.text(i, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 预测结果图已保存: {save_path}")
    
    plt.show()

def batch_predict(model, data_loader, device):
    """
    批量预测
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
    
    返回:
        预测结果和真实标签
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("🔍 正在进行批量预测...")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            
            # 收集结果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   已处理 {batch_idx + 1} 个批次")
    
    print(f"✅ 批量预测完成，共处理 {len(all_predictions)} 个样本")
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def calculate_metrics(predictions, labels):
    """
    计算预测指标
    
    参数:
        predictions: 预测结果
        labels: 真实标签
    
    返回:
        各种评估指标
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    
    # 分类报告
    report = classification_report(labels, predictions, 
                                 target_names=CIFAR10_CLASSES, 
                                 output_dict=True)
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }

def log_to_wandb(metrics, predictions, labels, probabilities, model_path):
    """
    将预测结果记录到wandb
    
    参数:
        metrics: 评估指标
        predictions: 预测结果
        labels: 真实标签
        probabilities: 预测概率
        model_path: 模型路径
    """
    # 记录指标
    wandb.log({
        "test_accuracy": metrics['accuracy'],
        "test_precision_macro": metrics['classification_report']['macro avg']['precision'],
        "test_recall_macro": metrics['classification_report']['macro avg']['recall'],
        "test_f1_macro": metrics['classification_report']['macro avg']['f1-score']
    })
    
    # 创建混淆矩阵图
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(metrics['confusion_matrix'], cmap='Blues')
    
    # 添加标签
    ax.set_xticks(range(len(CIFAR10_CLASSES)))
    ax.set_yticks(range(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel('预测类别')
    ax.set_ylabel('真实类别')
    ax.set_title('混淆矩阵')
    
    # 添加数值标签
    for i in range(len(CIFAR10_CLASSES)):
        for j in range(len(CIFAR10_CLASSES)):
            text = ax.text(j, i, metrics['confusion_matrix'][i, j],
                          ha="center", va="center", color="black")
    
    plt.tight_layout()
    
    # 上传到wandb
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close()
    
    # 记录模型文件
    model_artifact = wandb.Artifact(
        name=f"inference-model-{wandb.run.id}",
        type="model",
        description="用于推理的ResNet128模型"
    )
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)
    
    print("✅ 预测结果已记录到wandb")

def main():
    """
    主函数
    """
    print("🚀 开始ResNet128模型推理...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    
    # 模型路径
    model_path = "./models/best_model.pth"
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行训练脚本或检查模型路径")
        return
    
    # 初始化wandb（用于记录推理结果）
    wandb.init(
        project="resnet128-inference",
        name=f"inference-{os.path.basename(model_path)}",
        config={
            "model_path": model_path,
            "device": str(device),
            "num_classes": 10
        }
    )
    
    try:
        # 加载模型
        model, checkpoint = load_trained_model(model_path, device=device)
        
        # 示例：单张图像预测
        print("\n📸 单张图像预测示例:")
        print("注意：这里使用随机生成的图像作为示例")
        print("在实际使用中，您可以替换为真实的图像文件路径")
        
        # 创建随机图像进行演示（实际使用中替换为真实图像）
        random_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        random_image = Image.fromarray(random_image)
        
        # 预处理图像
        image_tensor, _ = preprocess_image_from_pil(random_image)
        
        # 进行预测
        predicted_class, confidence, all_probabilities = predict_single_image(
            model, image_tensor, device
        )
        
        print(f"预测类别: {CIFAR10_CLASSES[predicted_class]}")
        print(f"置信度: {confidence:.2%}")
        
        # 可视化结果
        visualize_prediction(
            random_image, 
            predicted_class, 
            confidence, 
            all_probabilities,
            save_path="prediction_result.png"
        )
        
        # 批量预测（如果有验证数据）
        print("\n📊 批量预测:")
        print("如果您有验证数据集，可以取消注释以下代码进行批量预测")
        
        # 示例代码（需要取消注释并修改）：
        """
        from torchvision.datasets import CIFAR10
        from torch.utils.data import DataLoader
        
        # 加载验证数据
        val_dataset = CIFAR10(root='./data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                    [0.2023, 0.1994, 0.2010])
                             ]))
        
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 批量预测
        predictions, labels, probabilities = batch_predict(model, val_loader, device)
        
        # 计算指标
        metrics = calculate_metrics(predictions, labels)
        
        print(f"测试准确率: {metrics['accuracy']:.4f}")
        
        # 记录到wandb
        log_to_wandb(metrics, predictions, labels, probabilities, model_path)
        """
        
        print("\n✅ 推理完成！")
        
    except Exception as e:
        print(f"❌ 推理过程中发生错误: {str(e)}")
        raise e
    
    finally:
        # 完成wandb运行
        wandb.finish()

def preprocess_image_from_pil(image, target_size=32):
    """
    从PIL图像预处理（用于演示）
    """
    # 调整图像大小
    image = image.resize((target_size, target_size))
    
    # 转换为张量并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    # 添加批次维度
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image

if __name__ == "__main__":
    main() 