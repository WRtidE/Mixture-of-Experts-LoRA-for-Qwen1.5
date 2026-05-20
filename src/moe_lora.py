"""
MoE-LoRA (Mixture of Experts LoRA) 核心模块

实现了一个可插拔的 MoE-LoRA 层，可以直接替换 HuggingFace 模型中的 nn.Linear 层。
每个专家有独立的 LoRA 低秩矩阵 (A, B)，通过可学习的路由器进行稀疏激活。

架构:
    input (batch, seq, d_in)
        |
        ├──> base_linear (冻结) ──> base_output
        |
        ├──> Router ──> routing_weights, expert_indices
        |
        └──> Experts (x K):
             for each activated expert k:
                 expert_output_k = B_k @ A_k @ input * scaling
             output = sum(routing_weight_k * expert_output_k)
        
        final = base_output + moe_lora_output

参考:
    - LoRA: https://arxiv.org/abs/2106.09685
    - MoE: https://arxiv.org/abs/1701.06538
    - Switch Transformers: https://arxiv.org/abs/2101.03961
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """MoE 路由器 / 门控网络

    根据输入 token 的 hidden state 决定激活哪些专家以及各自的权重。
    支持带噪声的 Top-k 门控（用于训练时的探索）。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        router_hidden_dim: Optional[int] = None,
        use_noisy_router: bool = True,
        noise_epsilon: float = 1e-2,
        init_scale: float = 0.02,
    ):
        """
        Args:
            hidden_dim: 输入隐藏维度
            num_experts: 专家总数
            top_k: 每个 token 激活的专家数
            router_hidden_dim: 路由器隐藏层维度（None 为单层）
            use_noisy_router: 是否注入噪声
            noise_epsilon: 噪声平滑系数
            init_scale: 初始化标准差缩放
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_noisy_router = use_noisy_router
        self.noise_epsilon = noise_epsilon

        if router_hidden_dim is not None:
            # 两层 MLP 路由器
            self.router = nn.Sequential(
                nn.Linear(hidden_dim, router_hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(router_hidden_dim, num_experts, bias=False),
            )
        else:
            self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        if use_noisy_router:
            if router_hidden_dim is not None:
                self.noise_router = nn.Sequential(
                    nn.Linear(hidden_dim, router_hidden_dim, bias=False),
                    nn.ReLU(),
                    nn.Linear(router_hidden_dim, num_experts, bias=False),
                )
            else:
                self.noise_router = nn.Linear(hidden_dim, num_experts, bias=False)

        self._init_weights(init_scale)

    def _init_weights(self, init_scale: float):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_scale)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)

        Returns:
            routing_weights: (batch_size, seq_len, top_k) 归一化后的专家权重
            expert_indices: (batch_size, seq_len, top_k) 选中的专家索引
            router_logits: (batch_size, seq_len, num_experts) 原始 logits（用于负载均衡）
        """
        # 计算路由 logits
        router_logits = self.router(hidden_states)  # (B, S, E)

        if self.use_noisy_router and self.training:
            noise_logits = self.noise_router(hidden_states)
            # 注入噪声: logits + StandardNormal() * softplus(noise_logits)
            noise = torch.randn_like(noise_logits)
            noisy_logits = router_logits + noise * F.softplus(noise_logits)
        else:
            noisy_logits = router_logits

        # Top-k 选择
        top_k_logits, expert_indices = torch.topk(
            noisy_logits, k=self.top_k, dim=-1
        )  # (B, S, top_k)

        # 对选中的 logits 做 softmax 归一化
        routing_weights = F.softmax(top_k_logits, dim=-1)  # (B, S, top_k)

        return routing_weights, expert_indices, router_logits

    def compute_load_balance_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算负载均衡损失

        参考 Switch Transformers 的 auxiliary loss:
            loss = num_experts * sum(f_i * P_i)

        其中 f_i 是分配给专家 i 的 token 比例，
        P_i 是路由器分配给专家 i 的平均概率。

        Args:
            router_logits: (B, S, E)
            expert_indices: (B, S, top_k)

        Returns:
            load_balance_loss: 负载均衡标量损失
            router_z_loss: 路由器 Z-loss（稳定训练用）
        """
        num_experts = self.num_experts

        # 路由概率 P_i: 每个专家被选中的平均 softmax 概率
        router_probs = F.softmax(router_logits, dim=-1)  # (B, S, E)
        # 展平 batch 和 seq 维度
        router_probs_flat = router_probs.view(-1, num_experts)  # (T, E)
        p_i = router_probs_flat.mean(dim=0)  # (E,)

        # 分配比例 f_i: 每个专家实际被分配到的 token 比例
        expert_indices_flat = expert_indices.view(-1, self.top_k)  # (T, top_k)
        f_i = torch.zeros(num_experts, device=router_logits.device)
        for e in range(num_experts):
            f_i[e] = (expert_indices_flat == e).sum().float()
        f_i = f_i / expert_indices_flat.numel()  # (E,)

        # 负载均衡损失
        load_balance_loss = num_experts * torch.sum(f_i * p_i)

        # Router Z-loss (鼓励 logits 保持在较小范围)
        router_z_loss = torch.mean(torch.square(router_logits))

        return load_balance_loss, router_z_loss


class MoELoRALinear(nn.Module):
    """MoE-LoRA 线性层

    用多个 LoRA 专家增强原始线性层。前向传播时：
        output = W @ x + sum_{k in top-k}(g_k * B_k @ A_k @ x) * (alpha / r)

    支持训练后的权重合并/分离操作。
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        lora_alpha: int = 32,
        num_experts: int = 8,
        top_k: int = 2,
        lora_dropout: float = 0.1,
        expert_dropout: float = 0.0,
        router_hidden_dim: Optional[int] = None,
        use_noisy_router: bool = True,
        noise_epsilon: float = 1e-2,
        init_scale: float = 0.02,
        fan_mode: str = "fan_in",
        bias: str = "none",
        init_lora_weights: str = "gaussian",
    ):
        """
        Args:
            base_linear: 原始 nn.Linear 层（将被冻结）
            r: LoRA 秩
            lora_alpha: LoRA 缩放系数
            num_experts: 专家数量
            top_k: 每个 token 激活的 top-k 专家
            lora_dropout: LoRA dropout
            expert_dropout: 专家级 dropout
            router_hidden_dim: 路由器隐藏层维度
            use_noisy_router: 使用带噪声路由器
            noise_epsilon: 噪声 epsilon
            init_scale: 路由器初始化缩放
            fan_mode: 初始化模式
            bias: bias 模式
            init_lora_weights: LoRA 权重初始化方式
        """
        super().__init__()

        # 基础层属性
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.num_experts = num_experts
        self.top_k = top_k
        self.scaling = lora_alpha / r
        self.fan_mode = fan_mode
        self.merged = False

        # 冻结原始权重
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        # LoRA 专家参数: 每个专家有独立的 A (down projection) 和 B (up projection)
        # A_experts: (num_experts, r, in_features)  -- 下投影
        # B_experts: (num_experts, out_features, r)  -- 上投影
        self.A_experts = nn.Parameter(
            torch.empty(num_experts, r, self.in_features)
        )
        self.B_experts = nn.Parameter(
            torch.empty(num_experts, self.out_features, r)
        )

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.expert_dropout = expert_dropout

        # 路由器
        self.router = Router(
            hidden_dim=self.in_features,
            num_experts=num_experts,
            top_k=top_k,
            router_hidden_dim=router_hidden_dim,
            use_noisy_router=use_noisy_router,
            noise_epsilon=noise_epsilon,
            init_scale=init_scale,
        )

        # Bias
        if bias == "all":
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        elif bias == "lora_only":
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

        # 初始化权重
        self._init_lora_weights(init_lora_weights)

    def _init_lora_weights(self, init_method: str):
        """初始化 LoRA 专家权重"""
        if init_method == "gaussian":
            # A: 高斯初始化, B: 零初始化（训练初期等价于原始模型）
            nn.init.kaiming_uniform_(self.A_experts, a=math.sqrt(5))
            nn.init.zeros_(self.B_experts)
        elif init_method == "kaiming":
            nn.init.kaiming_uniform_(self.A_experts, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.B_experts, a=math.sqrt(5))
        elif init_method == "pissa":
            # PiSSA 初始化: 用 SVD 初始化，保留主成分
            # 简化版：使用正交初始化
            nn.init.orthogonal_(self.A_experts)
            nn.init.orthogonal_(self.B_experts)
        else:
            raise ValueError(f"Unknown init method: {init_method}")

    def merge_weights(self):
        """将 MoE-LoRA 权重合并到基础权重中（推理时使用）

        合并所有专家权重的平均值到基础层。
        注意：这会丢失 MoE 的动态路由能力，仅用于简化部署。
        """
        if self.merged:
            return

        # 计算所有专家权重的平均值
        avg_B = self.B_experts.mean(dim=0)  # (out_features, r)
        avg_A = self.A_experts.mean(dim=0)  # (r, in_features)
        merged_delta = (avg_B @ avg_A) * self.scaling  # (out_features, in_features)
        self.base_linear.weight.data += merged_delta
        self.merged = True

    def unmerge_weights(self):
        """撤销权重合并"""
        if not self.merged:
            return

        avg_B = self.B_experts.mean(dim=0)
        avg_A = self.A_experts.mean(dim=0)
        merged_delta = (avg_B @ avg_A) * self.scaling
        self.base_linear.weight.data -= merged_delta
        self.merged = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, in_features)

        Returns:
            output: (batch_size, seq_len, out_features)
        
        Note:
            辅助损失 (load_balance_loss, router_z_loss) 存储在 self._last_losses 中，
            由外部 MoELoRAModel 在 forward 后收集。
        """
        # 基础线性变换
        base_output = self.base_linear(hidden_states)

        # 如果已合并，直接返回基础输出
        if self.merged:
            if self.bias is not None:
                base_output = base_output + self.bias
            self._last_losses = (None, None)
            return base_output

        # 路由
        routing_weights, expert_indices, router_logits = self.router(hidden_states)
        # routing_weights: (B, S, top_k), expert_indices: (B, S, top_k)

        batch_size, seq_len, _ = hidden_states.shape

        # 重塑 hidden_states 以便批量处理
        hidden_flat = hidden_states.view(-1, self.in_features)  # (B*S, d_in)

        # 应用 dropout
        hidden_dropped = self.lora_dropout(hidden_flat)  # (B*S, d_in)

        # 收集需要处理的 (token, expert) 对
        routing_flat = routing_weights.view(-1, self.top_k)  # (B*S, top_k)
        indices_flat = expert_indices.view(-1, self.top_k)  # (B*S, top_k)

        # 初始化输出
        moe_output = torch.zeros(
            batch_size * seq_len, self.out_features,
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        # 训练时应用专家级 dropout
        if self.training and self.expert_dropout > 0:
            dropout_mask = (
                torch.rand(self.num_experts, device=hidden_states.device) > self.expert_dropout
            )
        else:
            dropout_mask = torch.ones(self.num_experts, device=hidden_states.device)

        # ---- 逐专家计算 ----
        # 策略：遍历每个专家，找出所有路由到该专家的 (token, top_k_slot) 对，
        #       批量计算该专家的 LoRA 输出，再按路由权重加权累加到输出中。
        #       这样避免了为每个 token 单独计算，提升 GPU 利用率。
        for expert_idx in range(self.num_experts):
            # 专家级 dropout：训练时随机丢弃整个专家，增强鲁棒性
            if dropout_mask[expert_idx] == 0:
                continue

            # 找出 indices_flat 中所有等于当前 expert_idx 的位置
            # expert_mask: (B*S, top_k)，标记哪些 top_k 槽位选中了该专家
            expert_mask = (indices_flat == expert_idx)  # (B*S, top_k)
            # token_has_expert: (B*S,)，标记哪些 token 至少有一个槽位选中了该专家
            token_has_expert = expert_mask.any(dim=-1)  # (B*S,)

            # 如果没有 token 路由到该专家，跳过
            if not token_has_expert.any():
                continue

            # 收集所有路由到该专家的 token 的 hidden states
            expert_tokens = hidden_dropped[token_has_expert]  # (num_matched, d_in)

            # LoRA 双低秩分解：先降维 (d_in -> r)，再升维 (r -> d_out)
            A_out = F.linear(expert_tokens, self.A_experts[expert_idx])  # (num_matched, r)
            B_out = F.linear(A_out, self.B_experts[expert_idx])  # (num_matched, d_out)

            # 获取这些 token 在 top_k 中的路由权重和专家掩码
            token_weights = routing_flat[token_has_expert]  # (num_matched, top_k)
            token_expert_mask = expert_mask[token_has_expert]  # (num_matched, top_k)

            # 提取该专家的聚合权重：
            # 同一 token 可能在 top_k 中多次选中同一专家（虽然罕见），
            # 因此对多槽位权重求和得到该专家对该 token 的最终权重
            expert_weights = (token_weights * token_expert_mask.float()).sum(dim=-1)  # (num_matched,)

            # 加权累加到全局输出（in-place 加法，节省显存）
            moe_output[token_has_expert] += B_out * expert_weights.unsqueeze(-1)

        # LoRA 缩放：output *= alpha / r，控制 LoRA 分支的贡献幅度
        moe_output = moe_output * self.scaling

        # 将 (B*S, d_out) 恢复为 (B, S, d_out)
        moe_output = moe_output.view(batch_size, seq_len, self.out_features)

        # 合并输出
        output = base_output + moe_output

        if self.bias is not None:
            output = output + self.bias

        # 计算辅助损失并存储到实例属性中，供外部收集
        if self.training:
            self._last_losses = self.router.compute_load_balance_loss(
                router_logits, expert_indices
            )
        else:
            self._last_losses = (None, None)

        return output

    def get_last_losses(self):
        """返回最近一次前向传播的辅助损失 (load_balance_loss, router_z_loss)"""
        return getattr(self, "_last_losses", (None, None))


class MoELoRAModel(nn.Module):
    """将 MoE-LoRA 应用到整个 HuggingFace 模型的包装器

    自动识别并替换指定的线性层为 MoE-LoRA 层。
    支持从 transformers 模型加载和保存。
    """

    def __init__(
        self,
        model: nn.Module,
        moe_lora_config: "MoELoRAConfig",  # type: ignore
        target_modules: Optional[List[str]] = None,
    ):
        """
        Args:
            model: HuggingFace 预训练模型
            moe_lora_config: MoE-LoRA 配置
            target_modules: 要替换的目标模块名列表
        """
        super().__init__()
        from .config import MoELoRAConfig

        self.model = model
        self.moe_lora_config = moe_lora_config
        self.target_modules = target_modules or moe_lora_config.target_modules
        self.moe_lora_layers: Dict[str, MoELoRALinear] = nn.ModuleDict()

        # 冻结所有原始参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 应用 MoE-LoRA
        self._apply_moe_lora()

    def _apply_moe_lora(self):
        """遍历模型并替换目标模块
        
        工作原理：
        1. 遍历模型的所有命名子模块（named_modules）
        2. 对于名称末尾匹配 target_modules 的 nn.Linear 层
        3. 获取其父模块，用 MoELoRALinear 替换原来的 Linear 层
        4. 替换后的 MoE-LoRA 层同时保留了原始权重（冻结）和 LoRA 专家参数
        """
        replaced_count = 0
        for name, module in self.model.named_modules():
            # 取模块名的最后一段，如 "model.layers.0.self_attn.q_proj" -> "q_proj"
            module_name = name.split(".")[-1]
            if module_name not in self.target_modules:
                continue
            if not isinstance(module, nn.Linear):
                continue

            # 解析父模块路径和子模块属性名
            # 例如 name="model.layers.0.self_attn.q_proj"
            #   -> parent_name="model.layers.0.self_attn", child_name="q_proj"
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]

            # 获取父模块对象（顶级模块直接使用 self.model）
            if parent_name == "":
                parent = self.model
            else:
                parent = self.model.get_submodule(parent_name)

            # 创建 MoE-LoRA 层，将原 Linear 层作为冻结的基础层
            moe_lora_layer = MoELoRALinear(
                base_linear=module,
                r=self.moe_lora_config.r,
                lora_alpha=self.moe_lora_config.lora_alpha,
                num_experts=self.moe_lora_config.num_experts,
                top_k=self.moe_lora_config.top_k,
                lora_dropout=self.moe_lora_config.lora_dropout,
                expert_dropout=self.moe_lora_config.expert_dropout,
                router_hidden_dim=self.moe_lora_config.router_hidden_dim,
                use_noisy_router=self.moe_lora_config.use_noisy_router,
                noise_epsilon=self.moe_lora_config.noise_epsilon,
                init_scale=self.moe_lora_config.init_scale,
                fan_mode=self.moe_lora_config.fan_mode,
                bias=self.moe_lora_config.bias,
                init_lora_weights=self.moe_lora_config.init_lora_weights,
            )

            # 替换
            setattr(parent, child_name, moe_lora_layer)
            self.moe_lora_layers[name] = moe_lora_layer
            replaced_count += 1

        print(f"[MoELoRA] Replaced {replaced_count} linear layers with MoE-LoRA layers.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """前向传播，自动收集所有 MoE-LoRA 层的辅助损失

        Returns:
            outputs: 模型原始输出（包含 loss, logits 等）
            total_load_balance_loss: 所有 MoE-LoRA 层的负载均衡损失之和
            total_router_z_loss: 所有 MoE-LoRA 层的 router z-loss 之和
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        # 从所有 MoE-LoRA 层收集辅助损失
        total_load_balance_loss = torch.tensor(0.0, device=input_ids.device)
        total_router_z_loss = torch.tensor(0.0, device=input_ids.device)

        for layer in self.moe_lora_layers.values():
            lb_loss, z_loss = layer.get_last_losses()
            if lb_loss is not None:
                total_load_balance_loss = total_load_balance_loss + lb_loss
            if z_loss is not None:
                total_router_z_loss = total_router_z_loss + z_loss

        return outputs, total_load_balance_loss, total_router_z_loss

    def get_trainable_parameters(self):
        """获取所有可训练参数（仅 MoE-LoRA 参数）"""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
        return trainable_params

    def get_num_trainable_parameters(self) -> int:
        """返回可训练参数数量"""
        return sum(p.numel() for _, p in self.get_trainable_parameters())

    def merge_and_unload(self):
        """合并所有权重并返回基础模型（移除 MoE-LoRA 结构）"""
        for layer in self.moe_lora_layers.values():
            layer.merge_weights()
        return self.model

    def save_moe_lora(self, path: str):
        """仅保存 MoE-LoRA 权重（不包括冻结的基础模型权重）
        
        通过 requires_grad 标志筛选可训练参数，大幅减小检查点文件体积。
        """
        moe_lora_state = {
            name: param
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        torch.save(moe_lora_state, path)

    def load_moe_lora(self, path: str):
        """加载 MoE-LoRA 权重到当前模型结构
        
        使用 strict=False 允许部分加载，兼容不同配置的检查点。
        """
        moe_lora_state = torch.load(path, map_location="cpu")
        self.load_state_dict(moe_lora_state, strict=False)
