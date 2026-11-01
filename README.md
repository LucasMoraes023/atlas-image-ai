# ATLAS IMAGE AI — Creative Studio

Estúdio web de criação com IA generativa, editor de fotos e galeria local. O frontend usa React 18 + Vite; a API usa FastAPI e executa a inferência no Google Colab com GPU quando disponível.

## Estrutura

- `frontend/`: aplicação web (React/Vite), editor de fotos e galeria IndexedDB.
- `backend/api.py`: implementação oficial da API FastAPI.
- `colab/ATLAS_IMAGE_AI_COLAB.ipynb`: inicializa GPU, instala dependências, baixa o backend versionado e cria um túnel HTTPS temporário.
- `.github/workflows/project-validation.yml`: valida build do frontend, sintaxe Python e estrutura do notebook.

## Começar pelo celular

1. Publique a pasta `frontend` no Netlify. O `netlify.toml` da raiz define a pasta-base, o comando de build e a pasta publicada.
2. Abra `colab/ATLAS_IMAGE_AI_COLAB.ipynb` no Google Colab.
3. Em **Ambiente de execução → Alterar o tipo de ambiente de execução**, selecione GPU (T4 quando disponível).
4. Execute as células em ordem. O notebook baixa o backend oficial de `backend/api.py`, inicia a API e tenta criar um túnel HTTPS.
5. Abra no celular o link apresentado em **ABRA NO CELULAR**. O link contém uma chave temporária; não o compartilhe.
6. Mantenha a sessão do Colab ativa. Se ela expirar, execute novamente e use o novo link.

A primeira execução de cada modelo precisa baixar pesos. Tempo e disponibilidade dependem da GPU, da cota e do estado do Colab; não há garantia de sessão contínua nem de GPU gratuita.

## Recursos existentes

- Geração de imagens por texto e por imagem de referência (image-to-image, conforme o modelo).
- Modelos SDXL, Playground, RealVisXL, Animagine, SDXL-Turbo e FLUX.1 schnell.
- Estilos, proporções, seed, quantidade de imagens, passos e orientação.
- Melhoria de prompt e tradução opcional.
- Editor no navegador com brilho, contraste, saturação, rotação, espelhamento e texto.
- Galeria local no navegador (IndexedDB).

A disponibilidade real de cada modelo depende de memória e compatibilidade da GPU. FLUX.1 schnell é experimental em ambientes gratuitos e não oferece image-to-image nesta implementação.

## Segurança e privacidade

- A API exige o cabeçalho `X-Atlas-Key` nos endpoints de geração, consulta de tarefas, cancelamento e melhoria de prompt.
- A chave é gerada a cada sessão do Colab e transportada no fragmento `#` do link, não como parte do caminho enviado ao servidor.
- Não publique nem salve saídas do notebook que contenham o link de conexão ou a chave.
- Imagens de referência aceitam PNG, JPEG ou WebP, até 10 MB e 16 megapixels.
- O túnel público é temporário. A chave reduz acesso não autorizado, mas não substitui um serviço de hospedagem de produção.
- A galeria fica no armazenamento local do navegador; limpar os dados do navegador pode apagar as criações salvas.

## Desenvolvimento local

Frontend:

```bash
cd frontend
npm ci
npm run dev
```

Build de produção:

```bash
npm ci
npm run build
```

Backend: o arquivo `backend/requirements.txt` instala as dependências básicas da API. PyTorch com CUDA e Diffusers precisam ser instalados de acordo com a GPU/ambiente; o notebook do Colab prepara esse ambiente automaticamente.

Defina `ATLAS_KEY` com uma chave forte antes de iniciar a API. Opcionalmente, defina `ATLAS_ORIGINS` com os domínios permitidos separados por vírgula. Para desenvolvimento, o padrão CORS é permissivo; em hospedagem própria, configure os domínios explicitamente.

## Validação automática

Cada push/PR que altera frontend, backend ou notebook executa:
- `npm ci` e `npm run build`;
- compilação sintática de `backend/api.py`;
- validação do JSON do notebook e da referência ao backend versionado.

## Licenças dos modelos

Os pesos são mantidos por seus respectivos autores. Consulte a licença e os termos de uso de cada modelo no Hugging Face antes de uso comercial, redistribuição ou geração para terceiros. O ATLAS não concede direitos sobre os modelos de terceiros.
