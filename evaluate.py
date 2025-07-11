import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
from datetime import datetime
import os

def compute_gmean(y_true, y_pred):
    """
    计算G-mean: sqrt(recall * specificity)
    sensitivity = recall of positive class
    specificity = recall of negative class
    """
    # 少数类(0)的召回率
    recall = recall_score(y_true, y_pred, pos_label=0)
    
    # 多数类(1)的召回率
    specificity = recall_score(y_true, y_pred, pos_label=1)
    
    # 计算G-mean
    g_mean = np.sqrt(recall * specificity)
    
    return g_mean

def compute_metrics(y_true, y_pred):
    """计算F1分数和G-mean"""
    # F1-score (针对少数类)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # G-mean
    g_mean = compute_gmean(y_true, y_pred)
    
    # 准确率
    accuracy = (y_true == y_pred).sum() / len(y_true)
    
    # 计算每个类别的准确率
    class_0_acc = ((y_true == 0) & (y_pred == 0)).sum() / (y_true == 0).sum()
    class_1_acc = ((y_true == 1) & (y_pred == 1)).sum() / (y_true == 1).sum()
    
    return {
        'accuracy': accuracy,
        'class_0_acc': class_0_acc,
        'class_1_acc': class_1_acc,
        'f1_minority': f1[0],  # 少数类F1
        'f1_majority': f1[1],  # 多数类F1
        'f1_macro': f1.mean(),  # 宏平均F1
        'g_mean': g_mean
    }

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Minority (0)', 'Majority (1)'],
                yticklabels=['Minority (0)', 'Majority (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到 {save_path}")
    
    # 关闭图形以释放内存，不显示窗口
    plt.close()

def evaluate_model(model, test_loader, save_dir='./', dataset_name=None, training_ratio=None, rho=None, dataset_obj=None, run_number=None):
    """
    评估模型性能并计算相关指标
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        save_dir: 保存结果的目录
        dataset_name: 数据集名称
        training_ratio: 训练完成比例
        rho: 不平衡率
        dataset_obj: 数据集对象，用于获取样本数量统计
        run_number: 运行次数编号，用于文件命名
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            # 确保数据是正确的形状和类型
            if len(data.shape) == 3:  # (N, 28, 28)
                data = data.unsqueeze(1)  # 添加通道维度 -> (N, 1, 28, 28)
            # 修正通道顺序（如果需要）
            if data.shape[1] != 3 and data.shape[-1] == 3:
                data = data.permute(0, 3, 1, 2)  # NHWC -> NCHW
            data = data.float().to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    metrics = compute_metrics(all_labels, all_preds)
    
    # 打印结果
    print("\n===== 模型评估结果 =====")
    print(f"总体准确率: {metrics['accuracy']:.4f}")
    print(f"少数类准确率: {metrics['class_0_acc']:.4f}")
    print(f"多数类准确率: {metrics['class_1_acc']:.4f}")
    print(f"少数类F1-score: {metrics['f1_minority']:.4f}")
    print(f"多数类F1-score: {metrics['f1_majority']:.4f}")
    print(f"宏平均F1-score: {metrics['f1_macro']:.4f}")
    print(f"G-mean: {metrics['g_mean']:.4f}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取数据集统计信息
    train_positive_count = 0
    train_negative_count = 0
    test_positive_count = 0
    test_negative_count = 0
    
    if dataset_obj is not None:
        try:
            # 获取类别分布
            class_distribution = dataset_obj.get_class_distribution()
            train_positive_count = int(class_distribution['train'][0])  # 正类(标签0)数量
            train_negative_count = int(class_distribution['train'][1])  # 负类(标签1)数量
            test_positive_count = int(class_distribution['test'][0])    # 正类(标签0)数量
            test_negative_count = int(class_distribution['test'][1])    # 负类(标签1)数量
            
            print(f"训练集 - 正类样本数: {train_positive_count}, 负类样本数: {train_negative_count}")
            print(f"测试集 - 正类样本数: {test_positive_count}, 负类样本数: {test_negative_count}")
        except Exception as e:
            print(f"获取数据集统计信息时出错: {e}")
    
    # 准备DataFrame数据
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = {
        '评估时间': [current_time],
        '数据集名称': [dataset_name if dataset_name else 'Unknown'],
        '训练完成比例': [training_ratio if training_ratio is not None else 'Unknown'],
        '不平衡率rho': [rho if rho is not None else 'Unknown'],
        '训练集正类样本数': [train_positive_count],
        '训练集负类样本数': [train_negative_count],
        '测试集正类样本数': [test_positive_count],
        '测试集负类样本数': [test_negative_count],
        '总体准确率': [metrics['accuracy']],
        '少数类准确率': [metrics['class_0_acc']],
        '多数类准确率': [metrics['class_1_acc']],
        '少数类F1-score': [metrics['f1_minority']],
        '多数类F1-score': [metrics['f1_majority']],
        '宏平均F1-score': [metrics['f1_macro']],
        'G-mean': [metrics['g_mean']]
    }
    
    new_df = pd.DataFrame(new_data)
    
    # Excel文件路径
    excel_path = os.path.join(save_dir, 'evaluation_results.xlsx')
    
    # 检查是否存在现有文件
    if os.path.exists(excel_path):
        try:
            # 读取现有数据（包含标题行）
            existing_df = pd.read_excel(excel_path, header=0)
            # 合并数据
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"读取现有Excel文件时出错: {e}")
            print("将创建新文件")
            combined_df = new_df
    else:
        combined_df = new_df
    
    # 保存到Excel文件（包含标题行）
    try:
        combined_df.to_excel(excel_path, index=False, header=True)
        print(f"评估结果已保存到 {excel_path}")
        print(f"当前文件包含 {len(combined_df)} 条评估记录")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")
    
    # 绘制混淆矩阵
    # 生成带数据集名称、不平衡率、训练完成比例和序号的文件名
    dataset_str = dataset_name if dataset_name else 'Unknown'
    rho_str = f"rho{rho}" if rho is not None else 'rhoUnknown'
    ratio_str = f"{training_ratio}" if training_ratio is not None else 'Unknown'
    

    
    cm_filename = f'{dataset_str}_{rho_str}_训练完成比{ratio_str}_第{run_number}次.png'
    cm_path = os.path.join(save_dir, cm_filename)
    
    plot_confusion_matrix(all_labels, all_preds, save_path=cm_path)
    
    return metrics
