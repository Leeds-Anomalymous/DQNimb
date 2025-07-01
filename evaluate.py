import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import os

def compute_gmean(y_true, y_pred):
    """
    计算G-mean: sqrt(sensitivity * specificity)
    sensitivity = recall of positive class
    specificity = recall of negative class
    """
    # 少数类(0)的召回率
    recall_minority = recall_score(y_true, y_pred, pos_label=0)
    
    # 多数类(1)的召回率
    recall_majority = recall_score(y_true, y_pred, pos_label=1)
    
    # 计算G-mean
    g_mean = np.sqrt(recall_minority * recall_majority)
    
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
                xticklabels=['少数类(0)', '多数类(1)'],
                yticklabels=['少数类(0)', '多数类(1)'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到 {save_path}")
    
    plt.show()

def evaluate_model(model, test_loader, save_dir='./'):
    """
    评估模型性能并计算相关指标
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        save_dir: 保存结果的目录
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
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
    
    # 保存指标到文本文件
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("===== 模型评估结果 =====\n")
        f.write(f"总体准确率: {metrics['accuracy']:.4f}\n")
        f.write(f"少数类准确率: {metrics['class_0_acc']:.4f}\n")
        f.write(f"多数类准确率: {metrics['class_1_acc']:.4f}\n")
        f.write(f"少数类F1-score: {metrics['f1_minority']:.4f}\n")
        f.write(f"多数类F1-score: {metrics['f1_majority']:.4f}\n")
        f.write(f"宏平均F1-score: {metrics['f1_macro']:.4f}\n")
        f.write(f"G-mean: {metrics['g_mean']:.4f}\n")
    
    print(f"评估指标已保存到 {metrics_path}")
    
    # 绘制混淆矩阵
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, save_path=cm_path)
    
    return metrics
