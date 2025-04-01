import torch
import torch.nn as nn
import pandas as pd


class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.decoder = nn.Linear(feature_dim, feature_dim)

    def forward(self, smiles_features, image_features):
        # V, K from SMILES features
        V = self.v_proj(smiles_features)
        K = self.k_proj(smiles_features)

        # Q from image features
        Q = self.q_proj(image_features)

        # 计算注意力能量
        energy = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(128.0))
        attention = torch.softmax(energy, dim=-1)

        # 注意力加权和解码
        fusion = torch.matmul(attention, V)
        output = self.decoder(fusion)
        return output


def fuse_features():
    # 加载特征
    smiles_features = pd.read_csv('/root/BioICT/smiles_features.csv')
    image_features = pd.read_csv('/root/BioICT/molecular_features.csv')

    # 转换为tensor
    s_features = torch.tensor(smiles_features.iloc[:, 1:].values, dtype=torch.float32)
    i_features = torch.tensor(image_features.iloc[:, 1:].values, dtype=torch.float32)

    model = CrossAttentionFusion()
    fused_features = model(s_features, i_features)

    # 保存融合特征
    result_df = pd.DataFrame(fused_features.detach().numpy())
    result_df.to_csv('/root/BioICT/fused_features.csv', index=False)


if __name__ == "__main__":
    fuse_features()