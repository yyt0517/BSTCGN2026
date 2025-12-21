"""测试DataLoader作用域问题"""

def test_scope():
    # 创建变量
    train_loader = "train_loader_created"
    val_loader = "val_loader_created"
    test_loader = "test_loader_created"
    
    models_to_run = ['stgcn']
    
    # 在条件块中使用
    if 'cnn1d' in models_to_run:
        print(f"CNN1D: {train_loader}")
    
    if 'stgcn' in models_to_run:
        print(f"STGCN: {train_loader}")
        print(f"STGCN: {val_loader}")
        print(f"STGCN: {test_loader}")

if __name__ == '__main__':
    test_scope()
    print("✅ 测试通过！")

