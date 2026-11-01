# API do ATLAS IMAGE AI

A implementação oficial está em `api.py`. O notebook `colab/ATLAS_IMAGE_AI_COLAB.ipynb` baixa esse arquivo antes de iniciar o servidor, evitando manter uma cópia divergente dentro do notebook.

## Endpoints

| Método | Rota | Autenticação |
|---|---|---|
| GET | `/health` | Pública (diagnóstico básico) |
| POST | `/jobs` | `X-Atlas-Key` |
| GET | `/jobs/{id}` | `X-Atlas-Key` |
| POST | `/jobs/{id}/cancel` | `X-Atlas-Key` |
| POST | `/prompt` | `X-Atlas-Key` |

## Configuração

Defina `ATLAS_KEY` como segredo aleatório e forte. Opcionalmente, defina `ATLAS_ORIGINS` com uma lista de origens CORS separadas por vírgula. Em produção, não mantenha o curinga `*`.

Instale as dependências básicas:

```bash
pip install -r requirements.txt
```

PyTorch, Diffusers, Transformers e bibliotecas de inferência devem ser instalados separadamente, de acordo com a versão CUDA e a GPU. O notebook do Colab faz essa preparação.

## Validações e limites

- Prompts vazios, modelos, estilos e proporções inválidos são rejeitados.
- Imagens de referência aceitam PNG, JPEG e WebP, até 10 MB e 16 megapixels.
- O cabeçalho de autenticação é comparado em tempo constante.
- Erros internos não devolvem detalhes completos de exceções ao cliente.
- A fila limita tarefas pendentes e mantém o histórico de tarefas ativas.

## Observação

Esta API foi projetada para execução em sessão de GPU temporária do Colab. Não é um serviço de produção com disponibilidade contínua, armazenamento persistente ou escalabilidade multiusuário.
