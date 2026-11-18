# Mega-Sena Lab

Aplicativo web de **estudo de machine learning** aplicado ao histórico da Mega-Sena. Importa o `.xlsx` de resultados, treina modelos, **valida estatisticamente contra o acaso**, gera jogos para estudo e simula o ROI de apostas em um backtest.

> ⚠️ **Aviso importante:** a Mega-Sena é um sorteio aleatório e independente. Este projeto é **educacional**. Os modelos não superam o acaso, e o aplicativo foi feito justamente para demonstrar isso com testes estatísticos. Os jogos gerados **não aumentam a chance de ganhar** e não são recomendação de aposta.

---

## Funcionalidades

| Etapa | O que faz |
|---|---|
| **1. Importar** | Lê o `.xlsx` (colunas `Bola1`…`Bola6`; `Concurso`, `Data` e `Rateio 4/5/6 acertos` são opcionais) e valida os dados. |
| **2. Treinar** | Random Forest ou rede neural (MLP em janela de 20 concursos). Valida em concursos que ficaram fora do treino e informa se há evidência acima do acaso. |
| **3. Gerar previsões** | Define a quantidade de jogos (1 a 1.000), filtros de soma e paridade, intensidade do modelo e semente. Exporta em CSV ou XLSX. |
| **4. Backtest de ROI** | Simula os últimos N anos de apostas com os rateios reais e compara o modelo com estratégias aleatórias. |

## Metodologia

**Atributos** (por concurso e dezena, calculados **somente com concursos anteriores**, sem vazamento de dados):

- atraso (concursos desde a última saída) e razão atraso / atraso médio;
- frequência em janelas de 10, 30, 100 e 300 concursos e frequência total;
- **ciclo das dezenas** (se a dezena ainda falta no ciclo atual, tamanho do ciclo, dezenas faltantes);
- **quadrantes do volante** (volante 6×10 dividido em 4) e frequência recente por quadrante;
- paridade, linha, coluna, número da dezena e se saiu no concurso anterior.

**Validação:**

- *Holdout* temporal (os últimos N concursos nunca entram no treino).
- **AUC** comparado com a distribuição de AUC com rótulos embaralhados (teste de permutação, p95).
- Acertos médios no top-6 contra o esperado ao acaso (0,6), com teste t.
- Backtest com *walk-forward* (retreino a cada 100 concursos) e comparação de ROI com 40 simulações aleatórias.
- Teste de acertos observados × esperados (hipergeométrica e binomial) para jogos com 4+ acertos.

## Resultados obtidos com o histórico 1996–2026 (3.086 concursos)

| Teste | Resultado |
|---|---|
| Random Forest (holdout de 500 concursos) | AUC 0,499 (limite do acaso, p95: 0,510); acertos no top-6: 0,586 (esperado 0,600); p = 0,66 |
| Rede neural (MLP) | AUC 0,501; acertos no top-6: 0,626; p = 0,42 |
| P(sair \| atraso) | ≈ 10% para qualquer atraso, ou seja, a dezena não tem memória |
| Backtest de 5 anos (5 jogos/concurso, R$ 6,00) | ROI do modelo entre −73% e −93% conforme a semente; aleatório ≈ −94% |

Nenhum modelo apresentou evidência consistente de superar o acaso. Resultados pontuais favoráveis em um backtest (por exemplo, poucas quadras a mais que o esperado) **não se repetem ao trocar a semente**, o que indica sorte de amostragem. Por isso o app tem um campo de semente no backtest.

## Instalação

Requisitos: Python 3.10+.

```bash
git clone https://github.com/<seu-usuario>/megasena-lab.git
cd megasena-lab
pip install -r requirements.txt
python app.py
```

Abra `http://127.0.0.1:8000`. No Windows, também dá para usar `iniciar.bat`.

### Google Colab

```python
!pip -q install flask openpyxl
%run /content/megasena_app/colab_run.py
```

## Uso

1. Importe o arquivo de resultados (por exemplo, `Mega-Sena.xlsx`, exportado da Caixa).
2. Treine o modelo e confira o veredito estatístico.
3. Gere os jogos de estudo (escolha a quantidade e os filtros) e exporte.
4. (Opcional) Rode o backtest com várias sementes para ver a variação por sorte.

## Estrutura

```
megasena_app/
├── app.py               # backend Flask: dados, atributos, modelos, geração e backtest
├── templates/
│   └── index.html       # interface (HTML/CSS/JS puro, sem dependências externas)
├── requirements.txt
├── iniciar.bat          # atalho para Windows
├── colab_run.py         # inicialização no Google Colab
└── README.md
```

### API (resumo)

| Método | Rota | Descrição |
|---|---|---|
| POST | `/api/upload` | Importa o `.xlsx` |
| POST | `/api/train` | Treina e valida o modelo (job assíncrono) |
| POST | `/api/predict` | Gera os jogos |
| POST | `/api/backtest` | Simula o ROI (job assíncrono) |
| GET | `/api/job/<id>` | Status e resultado de um job |
| GET | `/api/export.xlsx` | Exporta os últimos jogos gerados |

## Limitações

- A rede neural é um MLP sobre janela de 20 concursos, **não uma LSTM**. Uma LSTM (PyTorch/TensorFlow) é um próximo passo possível, mas não há sinal sequencial a capturar.
- O preço da aposta (padrão R$ 6,00) é editável na tela. Confirme o valor vigente.
- O ROI usa os rateios históricos reais da planilha; em concursos sem ganhador da sena, não há retorno de sena.
- Aplicativo de uso local, de usuário único. Não foi feito para ser exposto na internet.

## Contribuindo

Sugestões e *pull requests* são bem-vindos, especialmente novos testes de aleatoriedade e uma implementação de LSTM.

## Licença

Escolha uma licença antes de publicar (por exemplo, [MIT](https://choosealicense.com/licenses/mit/)).

## Responsabilidade

Projeto sem qualquer vínculo com a Caixa Econômica Federal ou com a Mega-Sena. Jogue com responsabilidade: o retorno esperado de qualquer aposta é bem inferior ao valor apostado.
