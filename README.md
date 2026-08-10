AlphaGenome API benchmark helpers.

## Configuracao

O servidor le variaveis com prefixo `ALPHAGENOME_` a partir do ambiente ou do arquivo `.env`.

Defaults principais:

```env
ALPHAGENOME_HOST=0.0.0.0
ALPHAGENOME_PORT=8000
ALPHAGENOME_SEQUENCE_LEN=8000
ALPHAGENOME_BATCH_SIZE=1
ALPHAGENOME_WINDOW_SWEEP=false
```

`ALPHAGENOME_SEQUENCE_LEN` usa base 1000. Exemplos: `8000`, `16000`, `512000`.

`ALPHAGENOME_WINDOW_SWEEP=false` e o default seguro. Com ele desligado, o servidor rejeita requests cuja janela seja diferente de `ALPHAGENOME_SEQUENCE_LEN`.

Para liberar testes com varias janelas no mesmo servidor:

```env
ALPHAGENOME_WINDOW_SWEEP=true
```

## Rodar Server

```bash
python3 server.py
```

Exemplo com sweep liberado sem editar `.env`:

```bash
ALPHAGENOME_WINDOW_SWEEP=true python3 server.py
```

Exemplo fixando uma janela especifica:

```bash
ALPHAGENOME_SEQUENCE_LEN=512000 python3 server.py
```

## Rodar Client

Benchmark padrao compara chamadas sequenciais vs endpoint batch:

```bash
python3 client.py
```

Flags uteis:

```bash
python3 client.py --rounds 10 --num-variants 8 --batch-size 2 --window-size 8000
```

Pular chamadas sequenciais e testar apenas batch:

```bash
python3 client.py --batch-only --rounds 10 --num-variants 8 --batch-size 8
```

Trocar URL do server:

```bash
python3 client.py --base-url http://localhost:8000/v1
```

## Window Sweep

O sweep compara sequencial vs batch para janelas ate `512000` por default:

```bash
ALPHAGENOME_WINDOW_SWEEP=true python3 client.py --window-sweep --rounds 20 --num-variants 8 --batch-size 2
```

Janelas default:

```text
8000,16000,32000,64000,128000,256000,512000
```

Customizar janelas:

```bash
python3 client.py --window-sweep --window-sizes 8000,16000
```

Para testar apenas 8k em maquina limitada:

```bash
python3 client.py --window-sweep --window-sizes 8000 --rounds 20 --num-variants 2 --batch-size 2
```

Salvar em um CSV especifico:

```bash
python3 client.py --window-sweep --results-csv resultados_sweep.csv
```

Tambem existe override especifico do client:

```bash
ALPHAGENOME_CLIENT_WINDOW_SWEEP=true python3 client.py
```

Para o sweep funcionar de ponta a ponta, o server tambem precisa estar com:

```env
ALPHAGENOME_WINDOW_SWEEP=true
```

Se o server estiver com `ALPHAGENOME_WINDOW_SWEEP=false`, o client mostra `HTTP 422` nas janelas que nao batem com `ALPHAGENOME_SEQUENCE_LEN`.

## Interpretacao

Use `--window-sweep` para medir sequencial e batch em cada janela. O client imprime o tempo de cada rodada durante a execução e, no fim, mostra o resumo final.

A tabela inclui:

```text
janela, iters, vars, batch, seq melhor, seq media, batch melhor, batch media, speedup, status
```

O CSV inclui tambem os tempos individuais de cada rodada em `sequential_times_s` e `batch_times_s`.

Compare DGX vs A100 usando os mesmos valores de `rounds`, `num_variants`, `batch_size`, `requested_outputs` e `window_sizes`.
