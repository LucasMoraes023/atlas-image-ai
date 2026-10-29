# ATLAS IMAGE AI

Gerador de imagens por IA que roda **no navegador** (ONNX Runtime Web + WebGPU), sem API de inferência, sem chave e sem backend.

## Estado real

- Testado aqui: testes unitários (`npm test`) e build de produção (`npm run build`).
- **Não testado:** a geração de imagem em si. O ambiente de desenvolvimento não tinha GPU nem acesso ao Hugging Face. Nenhum teste foi feito em celular Android. O pipeline segue o exemplo oficial de SD-Turbo do ORT Web (guschmue/ort-webgpu), mas precisa ser validado no seu aparelho.

## Modelo

SD-Turbo em ONNX fp16 (`schmuell/sd-turbo-ort-web`), 512×512, 1 etapa, cerca de 2,4 GB baixados na primeira geração (ficam no cache do navegador). Exige Chrome com WebGPU e `shader-f16` (Android 12+ e GPU recente, vários GB de RAM). Em aparelhos modestos, o carregamento pode falhar por falta de memória.

**Licença:** confira a licença atual de `stabilityai/sd-turbo` e do repositório ONNX antes de qualquer uso comercial. As restrições de uso do modelo continuam valendo para o que você gerar.

## Controles

Reais: prompt, seed, presets (acrescentam texto visível ao prompt), galeria, download, exclusão, reuso.
Desabilitados, com explicação na tela: prompt negativo, orientação (CFG), etapas e resolução (o SD-Turbo roda sem CFG, em 1 etapa e 512×512 nesta versão).
O prompt enviado ao modelo é exatamente o do campo de texto. Não há lista de palavras bloqueadas.

## Privacidade

Prompts e imagens ficam no aparelho (IndexedDB). A única rede usada é o download dos arquivos do modelo e do tokenizador em huggingface.co. Limpar os dados do navegador apaga a galeria.

## Rodar

```
npm install
npm run dev      # desenvolvimento
npm test         # testes unitários
npm run build && npm run preview
```
WebGPU exige HTTPS ou localhost.

## Publicar pelo celular (grátis)

1. Crie um repositório no GitHub e envie os arquivos do projeto (sem `node_modules` e `dist`).
2. No Netlify: Add new site → Import from Git → escolha o repositório. O `netlify.toml` já define build e pasta `dist`.
3. Abra o endereço no Chrome do Android e toque em Gerar. Para desenvolver sem PC, use GitHub Codespaces (plano grátis) no navegador.

## Fora desta versão

Testes automatizados de geração/modelo, exportação e importação de configurações, múltiplos modelos, mais de 1 etapa, ícones PNG, escolha de resolução. Offline: aparece como "pronto" só quando modelo, tokenizador e service worker estão em cache.
