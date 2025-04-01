import torch
import torch.nn as nn
import math

import torch.nn.functional as F

def sequence_similarity(pred_seq, target_seq):
    """计算两个序列的相似度"""
    # 将logits转换为序列概率
    pred_probs = F.softmax(pred_seq, dim=-1)

    # 计算匹配长度
    min_len = min(pred_probs.size(1), target_seq.size(1))

    # 计算每个位置的相似度得分
    similarity_scores = torch.sum(
        pred_probs[:, :min_len] * F.one_hot(target_seq[:, :min_len], num_classes=pred_probs.size(-1)), dim=-1)

    return torch.mean(similarity_scores)

class ProteinTransformer(nn.Module):
    def __init__(self, smiles_vocab_size, protein_vocab_size, d_model=512, n_heads=8,
                 n_layers=6, d_ff=2048, dropout=0.1, max_length=1000):
        super().__init__()

        # Embeddings
        self.smiles_embedding = nn.Embedding(smiles_vocab_size, d_model)
        self.protein_embedding = nn.Embedding(protein_vocab_size, d_model)

        # Position Encoding
        self.register_buffer('pe', self._create_pe(max_length, d_model))

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        self.output_layer = nn.Linear(d_model, protein_vocab_size)
        self.max_length = max_length

    def _create_pe(self, max_length, d_model):
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, src, tgt=None, is_test=False):
        """
        src: [batch_size, seq_len] SMILES序列的token ids
        tgt: [batch_size * num_proteins, seq_len] 蛋白质序列的token ids
        """
        # Add positional encoding to embeddings
        src_emb = self.smiles_embedding(src) + self.pe[:, :src.size(1)]

        # Create padding masks
        src_pad_mask = (src == 0)

        if is_test:
            # 测试模式：生成序列
            src_output = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)
            return self.generate(src_output, src_pad_mask)

        if self.training:
            batch_size = src.size(0)
            # tgt是[batch_size * num_proteins, seq_len]的形状，因为每个源序列可能对应多个目标序列

            # 准备decoder输入（去掉最后一个token）
            tgt_input = tgt[:, :-1]
            # 准备目标输出（去掉第一个token）
            tgt_output = tgt[:, 1:]

            # 创建mask
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(src.device)
            tgt_pad_mask = (tgt_input == 0)

            # Decoder输入的embedding
            tgt_emb = self.protein_embedding(tgt_input) + self.pe[:, :tgt_input.size(1)]

            # Transformer的输出
            out = self.transformer(
                src_emb, tgt_emb,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
                tgt_mask=tgt_mask
            )

            # 得到logits [batch_size * num_proteins, seq_len, vocab_size]
            logits = self.output_layer(out)

            # 计算损失
            num_proteins_per_src = tgt.size(0) // batch_size  # 每个源序列对应的目标蛋白质数量
            total_loss = 0

            # 对每个源序列分别计算损失
            for i in range(batch_size):
                # 获取当前源序列对应的所有目标序列的输出和预测
                start_idx = i * num_proteins_per_src
                end_idx = (i + 1) * num_proteins_per_src
                current_logits = logits[start_idx:end_idx]  # [num_proteins, seq_len, vocab_size]
                current_targets = tgt_output[start_idx:end_idx]  # [num_proteins, seq_len]

                # 对每个目标序列计算交叉熵损失
                protein_losses = []
                for j in range(num_proteins_per_src):
                    loss = nn.functional.cross_entropy(
                        current_logits[j].view(-1, current_logits.size(-1)),
                        current_targets[j].view(-1),
                        ignore_index=0  # 忽略padding tokens
                    )
                    protein_losses.append(loss)

                # 取所有目标序列损失的最小值（因为我们希望至少与一个目标序列很相似）
                min_loss = torch.min(torch.stack(protein_losses))
                total_loss += min_loss

            # 返回平均损失
            return total_loss / batch_size

        else:
            # 验证模式
            return torch.tensor(0.0, device=src.device)

    def generate(self, memory, memory_pad_mask):
        batch_size = memory.size(0)
        device = memory.device

        print("\n=== 生成过程调试信息 ===")

        # 使用<START> token作为起始
        start_token = 1  # 假设1是<START>的索引，需要根据实际词汇表调整
        current_token = torch.full((batch_size, 1), start_token, device=device)
        generated_sequence = current_token

        # 降低温度参数使输出更确定性
        temperature = 1.2

        for step in range(self.max_length - 1):
            tgt_emb = self.protein_embedding(generated_sequence) + self.pe[:, :generated_sequence.size(1)]
            tgt_mask = self.transformer.generate_square_subsequent_mask(generated_sequence.size(1)).to(device)

            decoder_output = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_pad_mask
            )

            last_output = decoder_output[:, -1:]
            logits = self.output_layer(last_output)

            # 添加温度缩放
            scaled_logits = logits / temperature

            # 使用top_k采样来增加多样性
            top_k = 15
            top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)

            print(f"\nStep {step}")
            print(f"Top {top_k} probabilities: {probs[0].detach().cpu().numpy()}")
            print(f"Top {top_k} indices: {top_k_indices[0].detach().cpu().numpy()}")

            # 根据概率采样下一个token
            next_token_idx = torch.multinomial(probs.squeeze(), 1)
            next_token = top_k_indices.squeeze()[next_token_idx]

            print(f"Selected token: {next_token.item()}")

            # 将新token添加到序列中
            generated_sequence = torch.cat([generated_sequence, next_token.unsqueeze(0)], dim=1)

            # 检查是否生成了结束标记（假设2是<END>的索引）
            if next_token.item() == 2:  # 需要根据实际词汇表调整
                print("生成到END token，停止生成")
                break

            if step % 10 == 0:
                print(f"Current sequence length: {generated_sequence.size(1)}")
                print("Current sequence tokens:", generated_sequence[0].cpu().numpy())

        print("\n=== 生成过程结束 ===")
        return generated_sequence