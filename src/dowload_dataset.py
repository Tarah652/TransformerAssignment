"""
数据集下载脚本
用于下载 ChnSentiCorp 中文情感分析数据集
"""

import os
import sys

print("=" * 70)
print("  ChnSentiCorp 数据集下载工具")
print("=" * 70)

# 数据集信息
print("\n📊 数据集信息:")
print("   名称: ChnSentiCorp (中文情感分析语料库)")
print("   来源: 谭松波 - 中文情感挖掘语料")
print("   规模: ~12,000 条中文评论")
print("   内容: 酒店、书籍、电脑产品评论")
print("   Hugging Face: https://huggingface.co/datasets/seamew/ChnSentiCorp")
print("=" * 70)

# 检查 datasets 库
try:
    from datasets import load_dataset
    import pandas as pd

    print("\n✅ datasets 库已安装")
except ImportError:
    print("\n❌ 未安装 datasets 库")
    print("📦 正在安装...")
    os.system(f"{sys.executable} -m pip install datasets pandas")
    try:
        from datasets import load_dataset
        import pandas as pd

        print("✅ 安装成功")
    except:
        print("❌ 安装失败，请手动安装: pip install datasets pandas")
        sys.exit(1)

# 创建数据目录
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)
print(f"\n📁 数据目录: {data_dir}")

# 下载数据集
print("\n📥 开始下载数据集...")
print("⏳ 这可能需要几分钟，请耐心等待...")

try:
    # 下载训练集（限制 3000 条）
    print("\n1️⃣ 下载训练集...")
    dataset = load_dataset("seamew/ChnSentiCorp", split="train[:3000]")
    print(f"✅ 成功下载 {len(dataset)} 条训练数据")

    # 转换为 DataFrame
    df = pd.DataFrame(dataset)
    print(f"\n📊 数据预览:")
    print(df.head())

    # 保存为 CSV
    csv_path = os.path.join(data_dir, "ChnSentiCorp.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n💾 数据已保存到: {csv_path}")

    # 统计信息
    print(f"\n📈 数据统计:")
    print(f"   - 总条数: {len(df)}")
    print(f"   - 列名: {list(df.columns)}")
    if 'label' in df.columns:
        print(f"   - 正面评论: {(df['label'] == 1).sum()}")
        print(f"   - 负面评论: {(df['label'] == 0).sum()}")
    if 'text' in df.columns:
        print(f"   - 平均长度: {df['text'].str.len().mean():.1f} 字符")

    print("\n" + "=" * 70)
    print("✅ 数据集下载完成!")
    print("=" * 70)
    print("\n💡 下一步:")
    print("   运行 'python src/train.py' 开始训练")

except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n💡 备选方案:")
    print("   1. 检查网络连接")
    print("   2. 使用代理: export https_proxy=http://your-proxy:port")
    print("   3. 手动下载:")
    print("      访问 https://huggingface.co/datasets/seamew/ChnSentiCorp")
    print("      下载数据文件并放到 data/ 目录")
    print("   4. 使用备用数据: 代码会自动使用内置的中文语料")
    sys.exit(1)