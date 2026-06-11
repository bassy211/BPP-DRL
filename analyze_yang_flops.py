import torch
from thop import profile, clever_format
import sys

# 导入 Yang-DRL 项目中的模块
from acktr.arguments import get_args
from acktr.model import PUSNetBase

def analyze_yang_drl_complexity():
    # 1. 获取默认配置参数
    args = get_args()
    
    # 强制使用 CPU 进行测算，并明确启用 PUSNet
    args.use_cuda = False
    args.use_pusnet = True
    device = torch.device('cpu')

    print("="*50)
    print("⚙️ 正在加载网络配置参数 (Yang-DRL PUSNet 架构)...")
    print(f" - 容器尺寸 (w, l, h)    : {args.container_size}")
    print(f" - 隐层维度 (hidden_size): {args.hidden_size}")
    
    # 2. 计算输入的展平维度
    w, l, h = args.container_size
    area = w * l
    voxel_len = area * h       # 3D 体素输入长度 (10x10x10 = 1000)
    size_map_len = 3 * area    # 尺寸映射图长度 (3x10x10 = 300)
    
    # 在 PUSNetBase 的 forward 中，inputs 包含了 voxel 和 size_maps
    num_inputs = voxel_len + size_map_len
    
    # 3. 构造 Dummy Inputs
    batch_size = 1
    dummy_inputs = torch.randn(batch_size, num_inputs).to(device)
    
    # RNN 隐状态和掩码参数以对齐接口
    dummy_rnn_hxs = torch.zeros(batch_size, args.hidden_size).to(device)
    dummy_masks = torch.zeros(batch_size, 1).to(device)
    
    print(f"✅ 成功生成虚拟输入 Tensor，总展平长度为: {dummy_inputs.shape[1]}")
    print(f"   ├─ Voxel 状态长度: {voxel_len}")
    print(f"   └─ Size Maps 长度: {size_map_len}")

    # 4. 实例化核心网络 PUSNetBase
    print("⚙️ 正在构建 PUSNetBase (协同装拆网络) 模型...")
    model = PUSNetBase(num_inputs=num_inputs, recurrent=False, hidden_size=args.hidden_size, args=args).to(device)
    model.eval() # 切换到评估模式

    # 5. 调用 thop 进行统计
    print("⏳ 正在追踪计算图并计算 FLOPs...")
    try:
        # thop 追踪 forward 函数
        macs, params = profile(model, inputs=(dummy_inputs, dummy_rnn_hxs, dummy_masks), verbose=False)

        # 格式化输出
        macs_str, params_str = clever_format([macs, params], "%.3f")
        flops = macs * 2  # 1次乘加操作(MAC) 约等于 2次浮点运算(FLOP)
        _, flops_str = clever_format([0, flops], "%.3f")

        print("\n" + "📊 Yang-DRL (PUSNet 3D体素+双向协同) 复杂度报告".center(44))
        print("-" * 50)
        print(f"分支架构       : Conv3d (容器) + Conv2d (物品)")
        print(f"输出头数量     : 4 (Pack/Unpack Actor/Critic)")
        print(f"参数量 (Params): {params_str}")
        print(f"乘加数 (MACs)  : {macs_str}")
        print(f"运算量 (FLOPs) : {flops_str} (≈ 2 * MACs)")
        print("-" * 50 + "\n")
        
    except Exception as e:
        print(f"\n❌ 计算失败: {e}")

if __name__ == '__main__':
    # 将当前目录加入环境变量以防找不到包
    # sys.path.append('./')
    analyze_yang_drl_complexity()