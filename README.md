# ATLAS IMAGE AI (v2)

Site (React/Vite) + API FastAPI que roda na GPU gratuita do Google Colab. Funciona pelo celular.

## Como usar
1. Publique a pasta `frontend` (Netlify: o `netlify.toml` da raiz já configura; ou GitHub Pages com `npm run build` e a pasta `dist`).
2. Abra `colab/ATLAS_IMAGE_AI_COLAB.ipynb` no Colab, escolha **T4 GPU**, ajuste `FRONTEND_URL` na última célula e toque em **Executar tudo**.
3. A última célula mostra um **link de conexão**. Abra-o no celular: o site guarda a conexão e limpa o link da barra de endereço.
4. Se a sessão cair, execute a última célula de novo e abra o novo link.

## Segurança
O túnel é público, mas a API só responde com a chave aleatória gerada a cada execução (vai no `#` do link, que não é enviado a servidores). Não salve o notebook com a saída no GitHub: ela contém a chave. Cada execução gera uma chave nova.

## Transparência
O prompt enviado ao modelo é mostrado na tela. Estilo "Nenhum" não acrescenta nada; os outros acrescentam um texto exibido antes de gerar. A tradução PT→EN é opcional.

## Limites
Colab gratuito tem cota, sessões temporárias e GPU não garantida. FLUX.1 schnell é experimental no Colab gratuito. Confira a licença de cada modelo antes de uso comercial.

## Alternativas
- `colab/ATLAS_Image_AI.ipynb`: versão mais simples, tudo dentro do Colab (interface Gradio, sem site separado).
- Tag `v1-webgpu-wasm`: primeira versão, que roda o modelo no próprio navegador via WebGPU (exige celular potente).
