# SISTEMA-RH-DeepSeek

Projeto educativo para simulação de processos seletivos com machine learning e interface gráfica integrada. Este mini-projeto foi desenvolvido durante a disciplina ECOM031 - INTELIGÊNCIA ARTIFICIAL. Este mini-projeto combina modelagem preditiva, explicações de IA (SHAP) e uma interface interativa para análise de candidatos. Desenvolvido para estudos de sistemas de decisão interpretáveis em RH com experiência de usuário aprimorada. Projeto acadêmico sem aplicação em processos reais.

---

## EQUIPE:
1. Kauã Lessa Lima dos Santos
2. Luís Felipe Barros Pacheco
3. Diêgo de Araujo Correia
   
---

## PRÉ-REQUISITOS E INSTALAÇÃO

* Python 3.8 ou superior
* Bibliotecas essenciais:
  ```bash
  pip install pandas numpy lightgbm shap scikit-learn
  ```
* Tkinter (interface gráfica padrão do Python)
* Em sistemas Linux, caso necessário:
  ```bash
  sudo apt-get install python3-tk
  ```

---

## COMO CLONAR E EXECUTAR

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/KauaLessa/SISTEMA-RH-DeepSeek
   cd SISTEMA-RH-DeepSeek
   ```

2. **Ambiente virtual (recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Execute o sistema:**
   ```bash
   python main.py
   ```

---

## FUNCIONAMENTO PRINCIPAL

### Fluxo do Sistema
1. **Geração de Dados Sintéticos**  
   - 1000 candidatos simulados com características diversas
   - Variáveis: experiência, formação, competências técnicas e comportamentais
   - Balanceamento automático de features categóricas

2. **Modelo Preditivo Avançado**  
   - LightGBM multiclasse com otimização integrada
   - Sistema de pontuação técnica (0-100) e soft skills (1-5)
   - Tratamento especial para períodos de desemprego

3. **Interface Gráfica Interativa**  
   - Formulário unificado para entrada de dados
   - Controles intuitivos (sliders, comboboxes, checkboxes)
   - Visualização imediata de resultados com explicações SHAP

4. **Sistema Explicável**  
   - Detecção automática dos 3 principais fatores de decisão
   - Exibição de probabilidades por classe
   - Tradução de termos técnicos para linguagem de RH

---

## EXEMPLO DE INTERAÇÃO

```
Resultado da Classificação:
→ Decisão: Aprovado
→ Probabilidades:
- Aprovado: 68.50%
- Aprovado Parcialmente: 25.30%
- Reprovado: 6.20%

Fatores Decisivos:
• Pontuação Técnica: 89 (Impacto: 0.42)
• Soft Skills: 5 (Impacto: 0.38)
• Python: 1 (Impacto: 0.35)
```

---

## ESTRUTURA TÉCNICA

* Pipeline completo de machine learning:
  - Geração dinâmica de datasets sintéticos
  - Codificação automática de variáveis categóricas
  - Mapeamento inteligente de formações acadêmicas

* Interface gráfica profissional:
  - Validação em tempo real de entradas
  - Sistema de reset completo de formulário
  - Área de resultados com scroll e formatação

* Explicabilidade integrada:
  - Cálculo em tempo real de valores SHAP
  - Ranking de features impactantes
  - Exibição de impacto quantitativo

* Tratamento de erros robusto:
  - Verificação de tipos de dados
  - Validação de intervalos numéricos
  - Mensagens de erro contextualizadas

---

## IMPORTANTE

* **Dados simulados:** Não representam candidatos reais
* **Finalidade educacional:** Demonstração de conceitos de ML aplicado
* **Limitações técnicas:**
  - Modelo treinado com dados sintéticos
  - Intervalos fixos para pontuações
  - Não considera contexto subjetivo
* **Requisitos de uso:**
  - Preencher todos os campos do formulário
  - Valores numéricos dentro dos intervalos especificados
  - Selecionar pelo menos uma habilidade técnica

*Desenvolvido exclusivamente para fins acadêmicos - não utilizar em processos reais de seleção.*
