# Quickstart Superpowers (Copilot CLI)

Este guia mostra o caminho mais curto para ativar os skills do Superpowers em um projeto usando o GitHub Copilot CLI.

## 0. Instale o GitHub Copilot CLI

Se o Copilot CLI ainda nao estiver instalado, use um dos metodos abaixo:

### Windows (winget)

```powershell
winget install GitHub.Copilot
```
### Qualquer sistema (npm)

```bash
npm install -g @github/copilot
```

Valide a instalacao:

```bash
copilot --version
```

Faca login no Copilot CLI:

```bash
copilot login
```

## 1. Instale o plugin Superpowers

No terminal, execute:

```bash
copilot plugin marketplace add obra/superpowers-marketplace
copilot plugin install superpowers@superpowers-marketplace
```

## 2. Entre no projeto

Abra o diretório do seu projeto e inicie uma sessão do Copilot CLI nesse diretório.

Exemplo:

```bash
cd /caminho/do/seu/projeto
copilot
```

## 3. Ative os skills no início da sessão

Com o plugin instalado, o hook de SessionStart do Superpowers injeta automaticamente o contexto base (`using-superpowers`) quando a sessão começa.

Na prática, basta começar a conversa descrevendo sua tarefa. Exemplo:

```text
Quero implementar a feature X com TDD.
```

O agente passa a selecionar e aplicar os skills relevantes conforme o tipo de tarefa.

## 4. Verificação rápida

Teste com um pedido típico de fluxo Superpowers:

```text
Vamos começar com brainstorming antes de codar.
```

Se o agente responder guiando o fluxo (descoberta de contexto, desenho da solução, plano antes de implementação), os skills estão ativos.

## 5. Exemplo no terminal integrado (TDD)

No terminal integrado do VS Code, dentro do seu projeto:

```bash
copilot
```

Na sessão aberta, envie uma instrução explícita para TDD:

```text
Quero implementar o endpoint POST /users usando TDD. Comece no ciclo RED-GREEN-REFACTOR.
```

Resultado esperado:

- O agente propõe primeiro um teste que falha (RED).
- Depois implementa o mínimo para passar (GREEN).
- Em seguida sugere/refatora com segurança (REFACTOR).
- Entre etapas, ele valida com testes em vez de pular direto para código final.

## 6. Exemplo no terminal integrado (Debugging sistematico)

No terminal integrado do VS Code, dentro do projeto:

```bash
copilot
```

Na sessao aberta, peca um fluxo de investigacao guiada:

```text
Tenho um teste flaky em auth.spec.ts. Use debugging sistematico para achar a causa raiz antes de propor correcao.
```

Resultado esperado:

- O agente coleta evidencias (logs, reproducoes, hipoteses testaveis).
- Evita "chutes" e separa sintomas de causa raiz.
- So depois propoe a correcao e valida com reproducao + testes.

## Troubleshooting

- Se não funcionar de primeira, reinicie a sessão do Copilot CLI.
- Se o terminal integrado do VS Code mostrar "Cannot find GitHub Copilot CLI", abra um terminal externo (Windows Terminal, PowerShell ou CMD fora do VS Code) e execute o Copilot CLI por la.
- Confirme que o plugin foi instalado sem erros.
- Reinstale o plugin se necessário:

```bash
copilot plugin install superpowers@superpowers-marketplace
```

## Referências

- [Copilot CLI sessions in Visual Studio Code](https://code.visualstudio.com/docs/copilot/agents/copilot-cli#_create-a-copilot-cli-session)

