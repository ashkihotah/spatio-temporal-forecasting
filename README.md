# Experiments

## Overrides

### MoE

```bash
"model/cell_adapter@model.encoder_cell_adapter=HMoE,None" model.encoder_cell_adapter.cfg.moe_router.n_experts=4,8,16 model.encoder_cell_adapter.cfg.moe_router.top_k=1,2,3,4
```

```bash
"model/cell@model.encoder_cell=CMoE" model.encoder_cell.routing_cfg.top_k=1,2,3,4 "+overrides/CMoE=4Experts,8Experts,16Experts" model.encoder_cell.default_cell=MGAP-LSTM,GAP-LSTM model.encoder_cell.all_experts.GAP-LSTM.default_gnn=GCN,GCNSKIPCONN,GAT model.encoder_cell.all_experts.MGAP-LSTM.default_gnn=GCN,GCNSKIPCONN,GAT
```

```bash
model.encoder_cell.default_cell=GAP-LSTM "model.encoder_cell.cell.default_gnn=MoE-GNN" "+overrides/GMoE=4GNN,8GNN" "model.gnns.MoE-GNN.cfg.top_k=1,2,3,4" model.gnns.MoE-GNN.default_gnn=GCNSKIPCONN,GCN,GAT
```

### Attention

```bash
"overrides/attentions=Yes" "overrides/attentions/projections=Shared,HN" "overrides/attentions/pe_layers=None,RoPE,Sinusoidal"
```

### Cell Input Data Distributions

#### Different 
```bash
"overrides/cell-input-data-distribution=different" "model/postprocessing=STEDConvolution,STReshapedHead,STSharedHead" "framework.keras.losses.mae.cfg.encoder_outputs=null"
```

#### Same 
```bash
"overrides/cell-input-data-distribution=same" model.params.model.apply_enc_adapter_to_decoder_cell=true "framework.keras.losses.mae.cfg.encoder_outputs=null"
```

### Other
```bash
"model/preprocessing=None,NormFF" params.dataset.use_time_features=False,True
```

### Vanilla Models

```bash
python -m experiments.run "model/cell/cell@model.encoder_cell.cell=GRU"
```

```bash
python -m experiments.run "model/cell/cell@model.encoder_cell.cell=GAP-LSTM,MGAP-LSTM,GMoE-GAP-LSTM,GMoE-MGAP-LSTM" "model.encoder_cell.cell.default_gnn=GCN,GCNSKIPCONN,GAT" 
```

```bash
python -m experiments.run "model/cell/cell@model.encoder_cell.cell=LSTM" "model/cell_adapter@model.encoder_cell_adapter=SVDPP,GNNPP,None"
```

# All Models

- MGAP-LSTM-(GCN)
- MGAP-LSTM-(GCNSKIPCONN)
- GAP-LSTM-(GCN)
- GAP-LSTM-(GCNSKIPCONN)
- MGAP-LSTM-(GAT) with skipconn and norm as hpo
- GAP-LSTM-(GAT) with skipconn and norm as hpo

- GMoE-MGAP-LSTM-(GCN)
- GMoE-MGAP-LSTM-(GCNSKIPCONN)
- GMoE-GAP-LSTM-(GCN)
- GMoE-GAP-LSTM-(GCNSKIPCONN)
- GMoE-MGAP-LSTM-(GAT) with skipconn and norm as hpo
- GMoE-GAP-LSTM-(GAT) with skipconn and norm as hpo

- LSTM

- GNNPP-LSTM-(GCN)
- SVDPP-LSTM
- GRU

# Altieri Experiments

- NormFF, NoNorm (2)
- NoTimeFeatures, TimeFeatures (2)
- Attention
    - No
    - Yes (2)
        - HN-attention-NoPE
        - shared-attention-NoPE
- Data Distribution
    - EmbedingAR, NoExogFeat e NoShift (3)
        - Conv
        - STReshapedHead
        - STSharedHead
    - TargetAR, UseExogFeat, RShift e EncoderLoss

# My Experiments

- GMoE, None (2) (applicabile solo a GAP-LSTM)
- HMoEAdapter (HMoE), None (2)
- MoECell (CMoE), None (2)

# TODO

1. try on wind-nrel
2. try base keras hpo i.e. 
    1. batch_size: 32, 64, 128, 256, 512, 1024
    2. learning rate: 0.1, 0.01, 0.001, 0.0001
    3. epochs: 10, 25, 50
3. beijing gemini scalers
4. try attention pe layers
5. Try all other models: LSTM, GNN-LSTM, SVD-LSTM, GRU etc.


6. Quando vengono combinate diverse tecniche di MoE, l'aggregazione delle MoE loss è da rincontrollare poichè potrebbe non essere corretta. I moduli python da ricontrollare sono esattamente:
    1. `HMoeAdapter.py`: in particolare l'aggregazione `step_loss = step_loss + head_loss`
    2. `MoeCell.py`: in particolare l'aggregazione `step_loss = step_loss + expert_loss`
    3. `GAPLSTMWithMemoryCell.py`: in particolare l'aggregazione `step_loss = step_loss + mem_gnn_loss`. Qui l'aggregazione in realtà dovrebbe essere 
    ```python
    mgap_lstm_step_loss = loss_fx + loss_fh + loss_ix + loss_ih + loss_cx + loss_ch + loss_ox + loss_oh
    mgap_lstm_step_loss += loss_ih + loss_im + loss_gh + loss_gm + loss_oh + loss_om
    mgap_lstm_step_loss /= 14.0
    ```

La fix prioritaria è la (3.) relativa a `GAPLSTMWithMemoryCell.py` dato che senza quella non si possono runnare le GMoE. Quindi non appena finisce quest'ultima run per la CMoE, committiamo tutto così preserviamo tutti gli esperimenti fatti fin'ora. Fixiamo tutto e rilanciamo solo gli esperimenti sulla MoE.