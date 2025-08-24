#!/usr/bin/env python3
"""
ResNet128 训练启动脚本
提供用户友好的界面来启动训练

"""

import os
import sys
import subprocess
import time

def print_banner():
    """打印项目横幅"""
    print("=" * 60)
    print("🚀 ResNet128 深度学习训练项目")
    print("=" * 60)
    print("使用 Weights & Biases 进行实验跟踪和可视化")
    print("=" * 60)

def check_dependencies():
    """检查依赖包是否已安装"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        'torch', 'torchvision', 'wandb', 'numpy', 
        'matplotlib', 'PIL', 'tqdm', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_wandb_login():
    """检查wandb是否已登录"""
    print("\n🔑 检查Wandb登录状态...")
    
    try:
        import wandb
        # 尝试获取当前用户
        api = wandb.Api()
        user = api.default_entity
        print(f"✅ 已登录Wandb，用户: {user}")
        return True
    except Exception as e:
        print("❌ Wandb未登录或配置有误")
        print("请运行以下命令登录:")
        print("wandb login")
        return False

def create_directories():
    """创建必要的目录"""
    print("\n📁 创建项目目录...")
    
    directories = ['models', 'data', 'logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   ✅ 创建目录: {directory}")
        else:
            print(f"   📁 目录已存在: {directory}")

def show_training_options():
    """显示训练选项"""
    print("\n🎯 训练选项:")
    print("1. 开始完整训练 (推荐)")
    print("2. 快速测试训练 (5个epoch)")
    print("3. 自定义训练参数")
    print("4. 仅运行推理")
    print("5. 退出")
    
    while True:
        try:
            choice = input("\n请选择 (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            else:
                print("❌ 无效选择，请输入1-5之间的数字")
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            sys.exit(0)

def run_training(epochs=100, quick_test=False):
    """运行训练脚本"""
    print(f"\n🚀 开始训练...")
    
    if quick_test:
        print("⚡ 快速测试模式: 5个epoch")
        # 修改训练脚本中的epoch数量
        modify_epochs(5)
    
    try:
        # 运行训练脚本
        result = subprocess.run([sys.executable, 'resnet128_train.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 训练完成！")
            print("\n📊 训练结果:")
            print(result.stdout)
        else:
            print("❌ 训练过程中出现错误:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ 运行训练脚本时出错: {str(e)}")

def modify_epochs(epochs):
    """临时修改训练脚本中的epoch数量"""
    try:
        with open('resnet128_train.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换epoch数量
        content = content.replace('"epochs": 100', f'"epochs": {epochs}')
        
        with open('resnet128_train.py', 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"   📝 已临时修改训练轮次为 {epochs}")
        
    except Exception as e:
        print(f"   ⚠️ 修改训练参数失败: {str(e)}")

def run_inference():
    """运行推理脚本"""
    print("\n🔍 运行模型推理...")
    
    # 检查模型文件是否存在
    if not os.path.exists('./models/best_model.pth'):
        print("❌ 未找到训练好的模型文件")
        print("请先运行训练脚本")
        return
    
    try:
        result = subprocess.run([sys.executable, 'model_inference.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 推理完成！")
        else:
            print("❌ 推理过程中出现错误:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ 运行推理脚本时出错: {str(e)}")

def show_help():
    """显示帮助信息"""
    print("\n📚 使用说明:")
    print("=" * 40)
    print("1. 完整训练: 100个epoch，适合生产环境")
    print("2. 快速测试: 5个epoch，验证代码正确性")
    print("3. 自定义参数: 手动调整超参数")
    print("4. 推理测试: 使用训练好的模型进行预测")
    print("\n💡 提示:")
    print("- 首次运行会自动下载CIFAR-10数据集")
    print("- 训练过程会实时上传到wandb平台")
    print("- 最佳模型会自动保存到models目录")

def main():
    """主函数"""
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装所需包")
        return
    
    # 检查wandb登录
    if not check_wandb_login():
        print("\n❌ Wandb登录失败，请先登录")
        return
    
    # 创建目录
    create_directories()
    
    # 主循环
    while True:
        choice = show_training_options()
        
        if choice == '1':
            # 完整训练
            run_training(epochs=100, quick_test=False)
            break
            
        elif choice == '2':
            # 快速测试
            run_training(epochs=5, quick_test=True)
            break
            
        elif choice == '3':
            # 自定义参数
            print("\n🔧 自定义训练参数:")
            try:
                epochs = int(input("训练轮次 (默认100): ") or "100")
                lr = float(input("学习率 (默认0.001): ") or "0.001")
                batch_size = int(input("批次大小 (默认32): ") or "32")
                
                print(f"\n📝 自定义参数:")
                print(f"   训练轮次: {epochs}")
                print(f"   学习率: {lr}")
                print(f"   批次大小: {batch_size}")
                
                # 这里可以添加修改配置文件的代码
                print("⚠️ 自定义参数功能需要手动修改代码")
                print("请编辑 resnet128_train.py 文件中的配置部分")
                
            except ValueError:
                print("❌ 输入格式错误，使用默认参数")
            
            choice = input("\n是否开始训练? (y/n): ").strip().lower()
            if choice == 'y':
                run_training(epochs=epochs, quick_test=False)
            break
            
        elif choice == '4':
            # 运行推理
            run_inference()
            break
            
        elif choice == '5':
            # 退出
            print("\n👋 再见！")
            break
    
    print("\n🎉 项目运行完成！")
    print("📁 查看生成的文件:")
    print("   - models/: 训练好的模型")
    print("   - data/: 数据集")
    print("   - wandb.ai: 在线实验跟踪")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {str(e)}")
        print("请检查错误信息并重试") 