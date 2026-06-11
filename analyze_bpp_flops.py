import torch
from thop import profile, clever_format
import sys

# 导入项目中的模块
from acktr.arguments import get_args
from acktr.model import CNNPro

def analyze_bpp_drl_complexity():
    # 1. 获取默认配置参数
    # 如果在终端运行，这里会自动解析命令行参数（如 python script.py --pallet_size 10）
    args = get_args()
    
    # 强制使用 CPU 进行测算，脱离环境依赖
    args.use_cuda = False
    device = torch.device('cpu')

    print("="*50)
    print("⚙️ 正在加载网络配置参数...")
    print(f" - 容器尺寸 (container_size): {args.container_size}")
    print(f" - 托盘尺寸 (pallet_size)  : {args.pallet_size}")
    print(f" - 输入通道数 (channel)    : {args.channel}")
    print(f" - 隐层维度 (hidden_size)  : {args.hidden_size}")
    print(f" - 启用旋转 (enable_rotation): {args.enable_rotation}")
    
    # 2. 计算输入的展平维度
    # 在 CNNPro 的 forward 中，输入会被这样 reshape: 
    # x = inputs.reshape((-1, self.args.channel, self.args.pallet_size, self.args.pallet_size))
    num_inputs = args.channel * args.pallet_size * args.pallet_size
    
    # 3. 构造 Dummy Inputs
    batch_size = 1
    dummy_inputs = torch.randn(batch_size, num_inputs).to(device)
    
    # 虽然是非 RNN 结构，但 forward 签名中要求这两个参数
    dummy_rnn_hxs = torch.zeros(batch_size, args.hidden_size).to(device)
    dummy_masks = torch.zeros(batch_size, 1).to(device)
    
    print(f"✅ 成功生成虚拟输入 Tensor，形状为: {dummy_inputs.shape}")

    # 4. 实例化核心网络 CNNPro
    print("⚙️ 正在构建原版 CNNPro 模型...")
    # recurrent 设为 False，完全匹配主程序逻辑
    model = CNNPro(num_inputs=num_inputs, recurrent=False, hidden_size=args.hidden_size, args=args).to(device)
    model.eval() # 切换到评估模式

    # 5. 调用 thop 进行统计
    print("⏳ 正在追踪计算图并计算 FLOPs...")
    try:
        # thop 需要追踪 forward 函数，输入顺序必须匹配 (inputs, rnn_hxs, masks)
        macs, params = profile(model, inputs=(dummy_inputs, dummy_rnn_hxs, dummy_masks), verbose=False)

        # 格式化输出
        macs_str, params_str = clever_format([macs, params], "%.3f")
        flops = macs * 2  # 1次乘加操作(MAC) 约等于 2次浮点运算(FLOP)
        _, flops_str = clever_format([0, flops], "%.3f")

        print("\n" + "📊 原版 BPP-DRL (CNNPro) 复杂度分析报告".center(42))
        print("-" * 50)
        print(f"输入特征维度   : {num_inputs} (Channels:{args.channel} x W:{args.pallet_size} x H:{args.pallet_size})")
        print(f"参数量 (Params): {params_str}")
        print(f"乘加数 (MACs)  : {macs_str}")
        print(f"运算量 (FLOPs) : {flops_str} (≈ 2 * MACs)")
        print("-" * 50 + "\n")
        
    except Exception as e:
        print(f"\n❌ 计算失败: {e}")

if __name__ == '__main__':
    # 如果提示找不到 acktr 包，可以取消下面这行的注释，将当前目录加入环境变量
    # sys.path.append('./')
    analyze_bpp_drl_complexity()