

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_modals=3):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.scale = dim ** 0.5
        self.num_modals = num_modals

    def forward(self, modal_features):
        batch_size = modal_features[0].shape[0]
        query = self.query_proj(modal_features[0]).unsqueeze(1)
        keys = []
        values = []
        for feat in modal_features:
            keys.append(self.key_proj(feat).unsqueeze(1))
            values.append(self.value_proj(feat).unsqueeze(1))
        keys = torch.cat(keys, dim=1)
        values = torch.cat(values, dim=1)
        attn_weights = torch.matmul(query, keys.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        fused_features = torch.matmul(attn_weights, values).squeeze(1)
        return fused_features, attn_weights

class BandAttention(nn.Module):
    def __init__(self, in_dim, num_bands=NUM_BANDS):
        super().__init__()
        self.num_bands = num_bands
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, band_features):
        attn_weights = []
        for band_feat in band_features:
            weight = self.attention(band_feat)
            attn_weights.append(weight)
        attn_weights = torch.cat(attn_weights, dim=1)
        attn_weights = F.softmax(attn_weights, dim=1)
        band_features = torch.stack(band_features, dim=1)
        attn_weights = attn_weights.unsqueeze(-1)
        fused_band_feat = torch.sum(band_features * attn_weights, dim=1)
        return fused_band_feat, attn_weights

class lg_gnn(nn.Module):
    def __init__(self, dl, mmse_tensor=None, entropy_tensor=None, device='cuda',
                 ablation_mode=None):
        super(lg_gnn, self).__init__()
        self.dl = dl
        self.device = device
        self.node_ftr_dim = dl.node_ftr_dim
        self.num_bands = NUM_BANDS
        self.ablation_mode = ablation_mode

        self.node_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.node_ftr_dim, NODE_HIDDEN_DIM * 2),
                nn.ReLU(),
                nn.LayerNorm(NODE_HIDDEN_DIM * 2),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(NODE_HIDDEN_DIM * 2, NODE_HIDDEN_DIM)
            ) for _ in range(self.num_bands)
        ])
        for encoder in self.node_encoders:
            self._init_linear_layers(encoder)

        self.use_gat = ablation_mode != "no_gat"
        if self.use_gat:
            self.gnns = nn.ModuleList([
                GATConv(NODE_HIDDEN_DIM, FUSION_DIM, heads=1, concat=False)
                for _ in range(self.num_bands)
            ])
        else:
            self.gnns = nn.ModuleList([
                GCNConv(NODE_HIDDEN_DIM, FUSION_DIM)
                for _ in range(self.num_bands)
            ])

        self.gnn_norms = nn.ModuleList([
            nn.LayerNorm(FUSION_DIM) for _ in range(self.num_bands)
        ])
        self._init_gnn_layers()

        self.band_attention = BandAttention(FUSION_DIM, self.num_bands) if ablation_mode != "no_band_attn" else None
        self.band_embeddings = nn.Embedding(self.num_bands, NODE_HIDDEN_DIM) if ablation_mode != "no_band_emb" else None
        if self.band_embeddings is not None:
            nn.init.xavier_normal_(self.band_embeddings.weight)

        self.use_mmse = mmse_tensor is not None and ablation_mode != "no_mmse"
        self.mmse_encoder = None
        self.mmse_tensor = None
        if self.use_mmse:
            self.mmse_dim = mmse_tensor.shape[1]
            self.mmse_encoder = nn.Sequential(
                nn.Linear(self.mmse_dim, MMSE_HIDDEN_DIM),
                nn.ReLU(),
                nn.LayerNorm(MMSE_HIDDEN_DIM),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(MMSE_HIDDEN_DIM, FUSION_DIM)
            )
            self._init_linear_layers(self.mmse_encoder)
            self.mmse_tensor = mmse_tensor.to(device)

        self.use_entropy = entropy_tensor is not None and ablation_mode != "no_entropy"
        self.entropy_encoder = None
        self.entropy_tensor = None
        if self.use_entropy:
            self.entropy_dim = entropy_tensor.shape[1]
            self.entropy_encoder = nn.Sequential(
                nn.Linear(self.entropy_dim, ENTROPY_HIDDEN_DIM),
                nn.ReLU(),
                nn.LayerNorm(ENTROPY_HIDDEN_DIM),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(ENTROPY_HIDDEN_DIM, FUSION_DIM)
            )
            self._init_linear_layers(self.entropy_encoder)
            self.entropy_tensor = entropy_tensor.to(device)

        self.use_cross_modal = ablation_mode != "no_cross_modal"
        self.num_modals = 1
        if self.use_mmse: self.num_modals += 1
        if self.use_entropy: self.num_modals += 1
        self.cross_modal_attn = CrossModalAttention(FUSION_DIM, self.num_modals) if self.use_cross_modal else None

        self.classifier = nn.Sequential(
            nn.Linear(FUSION_DIM, FUSION_DIM),
            nn.ReLU(),
            nn.LayerNorm(FUSION_DIM),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(FUSION_DIM, 2)
        )
        self._init_linear_layers(self.classifier)

        self.use_mi_loss = ablation_mode != "no_mi_loss"
        if self.use_mi_loss:
            self.mi_projector = nn.Sequential(
                nn.Linear(FUSION_DIM, 8),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(8, 4)
            )
            self._init_linear_layers(self.mi_projector)

        self.gnn_weight = nn.Parameter(torch.ones(1))
        self.mmse_weight = nn.Parameter(torch.ones(1)) if self.use_mmse else None
        self.entropy_weight = nn.Parameter(torch.ones(1)) if self.use_entropy else None

        self.label_smoothing = 0.0 if ablation_mode == "no_label_smoothing" else LABEL_SMOOTHING
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.temp_scale = 1.0 if ablation_mode == "no_temp_scaling" else TEMP_SCALE

    def _init_linear_layers(self, modules):
        if isinstance(modules, nn.Sequential):
            for m in modules:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
        elif isinstance(modules, list):
            for m in modules:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
        elif isinstance(modules, nn.Linear):
            nn.init.xavier_normal_(modules.weight)
            if modules.bias is not None:
                nn.init.constant_(modules.bias, 0.0)

    def _init_gnn_layers(self):
        for gnn in self.gnns:
            gnn_layers = []
            if hasattr(gnn, 'lin'):
                gnn_layers.append(gnn.lin)
            else:
                if hasattr(gnn, 'lin_src'):
                    gnn_layers.append(gnn.lin_src)
                if hasattr(gnn, 'lin_dst'):
                    gnn_layers.append(gnn.lin_dst)
            if hasattr(gnn, 'lin'):
                gnn_layers.append(gnn.lin)
            self._init_linear_layers(gnn_layers)

    def compute_mi_loss(self, z1, z2):
        batch_size = z1.shape[0]
        z1 = F.layer_norm(z1, z1.shape[1:])
        z2 = F.layer_norm(z2, z2.shape[1:])
        p1 = self.mi_projector(z1)
        p2 = self.mi_projector(z2)
        sim_matrix = torch.mm(p1, p2.t()) / np.sqrt(4)
        labels = torch.arange(batch_size).to(self.device)
        loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)
        return loss / 2.0

    def forward(self, raw_features, labels=None, indices=None):
        batch_band_features = []
        for band_idx in range(self.num_bands):
            band_graphs = [sample_bands[band_idx] for sample_bands in raw_features]
            band_feats = []
            for graph in band_graphs:
                graph = graph.to(self.device)
                node_ftr = self.node_encoders[band_idx](graph.x)
                if self.band_embeddings is not None:
                    band_emb = self.band_embeddings(torch.tensor(band_idx, device=self.device))
                    node_ftr = node_ftr + band_emb.unsqueeze(0)
                node_ftr = F.layer_norm(node_ftr, node_ftr.shape[1:])
                if self.use_gat:
                    gnn_out = self.gnns[band_idx](node_ftr, graph.edge_index, edge_attr=graph.edge_attr)
                else:
                    gnn_out = self.gnns[band_idx](node_ftr, graph.edge_index)
                gnn_out = self.gnn_norms[band_idx](gnn_out)
                gnn_out = F.relu(gnn_out)
                gnn_out = F.dropout(gnn_out, p=DROPOUT_RATE, training=self.training)
                batch = torch.zeros(gnn_out.shape[0], dtype=torch.long).to(self.device)
                graph_feature = global_mean_pool(gnn_out, batch=batch)
                band_feats.append(graph_feature)
            band_feats = torch.cat(band_feats, dim=0)
            batch_band_features.append(band_feats)

        if self.band_attention is None:
            fused_gnn_feat = torch.stack(batch_band_features, dim=1).mean(dim=1)
        else:
            fused_gnn_feat, _ = self.band_attention(batch_band_features)
        fused_gnn_feat = fused_gnn_feat * self.gnn_weight

        mmse_features = None
        if self.use_mmse:
            if indices is not None:
                batch_mmse = self.mmse_tensor[indices]
            else:
                batch_mmse = self.mmse_tensor[:len(raw_features)]
            mmse_tensor_norm = F.layer_norm(batch_mmse, batch_mmse.shape[1:])
            mmse_features = self.mmse_encoder(mmse_tensor_norm) * self.mmse_weight

        entropy_features = None
        if self.use_entropy:
            if indices is not None:
                batch_entropy = self.entropy_tensor[indices]
            else:
                batch_entropy = self.entropy_tensor[:len(raw_features)]
            entropy_tensor_norm = F.layer_norm(batch_entropy, batch_entropy.shape[1:])
            entropy_features = self.entropy_encoder(entropy_tensor_norm) * self.entropy_weight

        modal_features = [F.layer_norm(fused_gnn_feat, fused_gnn_feat.shape[1:])]
        if self.use_mmse:
            modal_features.append(F.layer_norm(mmse_features, mmse_features.shape[1:]))
        if self.use_entropy:
            modal_features.append(F.layer_norm(entropy_features, entropy_features.shape[1:]))

        if self.use_cross_modal:
            fused_features, _ = self.cross_modal_attn(modal_features)
        else:
            fused_features = torch.cat(modal_features, dim=1)
            if fused_features.shape[1] != FUSION_DIM:
                fused_features = nn.Linear(fused_features.shape[1], FUSION_DIM).to(self.device)(fused_features)

        fused_features = F.relu(fused_features)
        fused_features = F.layer_norm(fused_features, fused_features.shape[1:])
        logits = self.classifier(fused_features) / self.temp_scale
        total_loss = 0.0

        if labels is not None:
            cls_loss = self.criterion(logits, labels)
            mi_loss = 0.0
            if self.use_mi_loss:
                if self.use_mmse:
                    mi_loss += self.compute_mi_loss(fused_gnn_feat, mmse_features) * 0.05
                if self.use_entropy:
                    mi_loss += self.compute_mi_loss(fused_gnn_feat, entropy_features) * 0.05
            total_loss = cls_loss + mi_loss
        else:
            total_loss = torch.tensor(0.0).to(self.device)
        return logits, total_loss

    def build_sample_graph(self, embeddings, indices):
        if indices is None:
            indices = np.arange(len(embeddings))
        return np.ones((len(indices), len(indices)))

