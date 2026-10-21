# Experiments

## Overrides

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
"overrides/cell-input-data-distribution=same" "framework.keras.losses.mae.cfg.encoder_outputs=null" model.params.model.apply_enc_adapter_to_decoder_cell=true
```

### Other
```bash
"model/preprocessing=None,NormFF" params.dataset.use_time_features=False,True
```

## Vanilla Models

```bash
python -m experiments.run "model/cell/cell@model.encoder_cell.cell=GRU"
```

```bash
python -m experiments.run "model/cell/cell@model.encoder_cell.cell=GAP-LSTM,MGAP-LSTM,GMoE-GAP-LSTM,GMoE-MGAP-LSTM" "model.encoder_cell.cell.default_gnn=GCN,GCNSKIPCONN,GAT" 
```

```bash
python -m experiments.run "model/cell/cell@model.encoder_cell.cell=LSTM" "model/cell_adapter@model.encoder_cell_adapter=SVDPP,GNNPP,None"
```

# All configs

all vanilla models:

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

la tabella sarà

<table>
    <thead>
        <!-- Macro Columns (Parent Headers) -->
        <tr>
            <th colspan="1">Models</th>
            <th colspan="8">Encoder-Decoder Input Data Distribution</th>
        </tr>
        <tr>
            <th colspan="1"></th>
            <th colspan="4">Different</th>
            <th colspan="4">Same</th>
        </tr>
        <!-- Micro Columns (Child Headers) -->
        <tr>
            <th></th>
            <th>Min</th>
            <th>Avg</th>
            <th>Median</th>
            <th>Max</th>
            <th>Min</th>
            <th>Avg</th>
            <th>Median</th>
            <th>Max</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>LSTM</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        </tr>
        <tr>
        <td>SVDPP-LSTM</td>
        <td>0.8</td>
        <td>0.5</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        </tr>
    </tbody>
</table>

<table>
    <thead>
        <!-- Macro Columns (Parent Headers) -->
        <tr>
            <th colspan="1">Models</th>
            <th colspan="32">Attentions</th>
        </tr>
        <tr>
            <th colspan="1"></th>
            <th colspan="4">None</th>
            <th colspan="4">HN-NoPE</th>
            <th colspan="4">shared-NoPE</th>
            <th colspan="4">shared-SinPE</th>
        </tr>
        <!-- Micro Columns (Child Headers) -->
        <tr>
            <th></th>
            <th>Min</th>
            <th>Avg</th>
            <th>Median</th>
            <th>Max</th>
            <th>Min</th>
            <th>Avg</th>
            <th>Median</th>
            <th>Max</th>
            <th>Min</th>
            <th>Avg</th>
            <th>Median</th>
            <th>Max</th>
            <th>Min</th>
            <th>Avg</th>
            <th>Median</th>
            <th>Max</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>LSTM</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        </tr>
        <tr>
        <td>SVDPP-LSTM</td>
        <td>0.8</td>
        <td>0.5</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        <td>0.7</td>
        <td>0.6</td>
        </tr>
    </tbody>
</table>

Al posto di far vedere le tabelle è meglio far vedere i boxplot. I boxplot li puoi visualizzare sia in 3d sia in 2d ma rispondono a domande diverse. Per poterli visualizzare in 2d devi calcolare min, avg, median e max anche lungo i modelli vanilla ottendo un boxplot (configs x metric). In 3d avresti (models x configs x metric). Quelli in 2d rispondono alla domanda: Qual'è la config migliore indipendentemente dalla variante LSTM? Quelli in 3d rispondono sia alla domanda di prima sia alla domanda: Qual'è la combinazione config-modello migliore per risolvere questo task in questo dataset?

Min avg median e max sono calcolati fissando i modelli e aggregando su tutte le combinazioni (n. combinazioni totali sono 192) di:

- NormFF, NoNorm (2)
- NoTimeFeatures, TimeFeatures (2)

- HMoEAdapter, None (2)
- MoECell, None (2)

- NoAttention, HN-attention-NoPE, shared-attention-NoPE, shared-attention-SinPE (4)
- (EmbedingAR, NoExogFeat e NoShift), (TargetAR, UseExogFeat e RShift) (2)

- NoEncoderLoss, EncoderLoss (2)
- Conv, STReshapedHead, STSharedHead (3) (applicabili solo se scegli (EmbedingAR, NoExogFeat e NoShift))
- GMoE, None (2) (applicabile solo a GAP-LSTM)

n. Run: 
- 10 (modelli base) x 192 (hp sets) = 1920

- 1920 x 5 (iperparametri in hpo) = 9600
- 9600 x 30 s (run time) = 288.000 s = 4800 min = 80 h = 3 gg

- 3 gg x 2 dataset = 6 gg