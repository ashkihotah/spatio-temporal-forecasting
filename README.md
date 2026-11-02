# Experiments

## Overrides

### GNN

The following overrides are valid only with the following overrides: `"model/cell/cell@model.encoder_cell.cell=GAP-LSTM,MGAP-LSTM"`

```
"model.encoder_cell.cell.default_gnn=GCN,GCNSKIPCONN,GAT"

"++model.encoder_cell.cell.default_gnn=MoE-GNN" "+overrides/GMoE=6GNN,9GNN" model.moe_routers.default_args.top_k=1,2,3,4
```

### HMoE

```bash
"model/cell_adapter@model.encoder_cell_adapter=HMoE" model.moe_routers.default_args.n_experts=8,9,15,16 model.moe_routers.default_args.top_k=1,2,3,4
```

### Attention

```bash
"overrides/attentions=Yes" "overrides/attentions/projections=Shared,HN" "overrides/attentions/pe_layers=RoPE,Sinusoidal"
```

### Cell Input Data Distributions

#### Different 
```bash
"overrides/cell-input-data-distribution=different" "model/postprocessing=STEDConvolution,STReshapedHead,STSharedHead" "framework.keras.losses.mae.cfg.encoder_outputs=null" apply_enc_adapter_to_decoder_cell=true
```

#### Same 
```bash
"overrides/cell-input-data-distribution=same" model.params.model.apply_enc_adapter_to_decoder_cell=true "framework.keras.losses.mae.cfg.encoder_outputs=null"
```

### Other
```bash
"model/preprocessing=None,NormFF" params.dataset.use_time_features=True,False dataset=wind-nrel,pv-italy,beijing-multisite-airquality
```

### RNN Cell

```
"model/cell/cell@model.encoder_cell.cell=GRU"

"model/cell/cell@model.encoder_cell.cell=GAP-LSTM,MGAP-LSTM" 

"model/cell/cell@model.encoder_cell.cell=LSTM" "model/cell_adapter@model.encoder_cell_adapter=SVDPP,GNNPP,None"

"model/cell@model.encoder_cell=CMoE" "+overrides/CMoE=6E,9E,M6E,M9E"
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

- try base keras hpo i.e. 
    - batch_size: 32, 64, 128, 256, 512, 1024
    - learning rate: 0.1, 0.01, 0.001, 0.0001
    - epochs: 10, 25, 50
- beijing gemini scalers
- Try all other models: LSTM, GNN-LSTM, SVD-LSTM, GRU etc.
- rerunna tutte le GMoE da 6E
- rerunna tutte le GMoE con filtro `"Shazeer-MMP-GMoE-|GAT|-|GCN|-|GCNSKIPCONN|-LSTM" in run.experiment` in pv-italy