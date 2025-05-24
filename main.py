# -*- coding: utf-8 -*-
# Instale as dependências: pip install pandas numpy lightgbm shap

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import lightgbm as lgb
import shap
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# ======================================
# 1. Geração de Dados Sintéticos para Treinamento
# ======================================
np.random.seed(42)
n_samples = 1000

dados = {
    'experiencia_anos': np.random.randint(0, 10, n_samples),
    'formacao': np.random.choice(['Graduação', 'Mestrado', 'Bootcamp'], n_samples),
    'score_tecnico': np.random.normal(70, 15, n_samples).clip(0, 100).astype(int),
    'score_softskills': np.random.randint(1, 6, n_samples),
    'tempo_ultimo_emprego': np.random.exponential(6, n_samples).astype(int),
    'linguagens': [np.random.choice(['Python', 'Java', 'JavaScript', 'C++', 'SQL'], 
                   np.random.randint(0, 3)) for _ in range(n_samples)]
}

df = pd.DataFrame(dados)

# Codificar variáveis categóricas
mlb = MultiLabelBinarizer()
linguagens_encoded = pd.DataFrame(mlb.fit_transform(df['linguagens']), columns=mlb.classes_)
df = pd.concat([df.drop('linguagens', axis=1), linguagens_encoded], axis=1)

# Criar variável target
conditions = [
    (df['score_tecnico'] >= 80) & (df['score_softskills'] >= 4),
    ((df['score_tecnico'] >= 65) & (df['score_tecnico'] < 80)) | 
    ((df['score_softskills'] >= 3) & (df['score_softskills'] < 4)),
    (df['score_tecnico'] < 65) | (df['score_softskills'] < 3)
]
choices = ['Aprovado', 'Aprovado Parcialmente', 'Reprovado']
df['decisao'] = np.select(conditions, choices, default='Reprovado')

# Preparar dados para treino
features = df.columns.drop(['decisao'])
X = df[features]
y = df['decisao']

# Codificar formações
X['formacao'] = X['formacao'].map({'Graduação': 0, 'Mestrado': 1, 'Bootcamp': 2})

# Treinar modelo
model = lgb.LGBMClassifier(num_class=3, objective='multiclass')
model.fit(X, y)

# Preparar SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ======================================
# 2. Interface Gráfica Tkinter
# ======================================
class AplicativoRH:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de RH Inteligente")
        self.root.geometry("600x500")
        self.criar_formulario()
        
    def criar_formulario(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Widgets do formulário
        ttk.Label(main_frame, text="Anos de Experiência:").grid(row=0, column=0, sticky=tk.W)
        self.experiencia = ttk.Entry(main_frame)
        self.experiencia.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Formação:").grid(row=1, column=0, sticky=tk.W)
        self.formacao = ttk.Combobox(main_frame, values=["Graduação", "Mestrado", "Bootcamp"])
        self.formacao.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Habilidades Técnicas:").grid(row=2, column=0, sticky=tk.W)
        self.habilidades_frame = ttk.Frame(main_frame)
        self.habilidades_frame.grid(row=2, column=1, sticky=tk.W)
        self.checkboxes = {}
        for i, lang in enumerate(['Python', 'Java', 'JavaScript', 'C++', 'SQL']):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(self.habilidades_frame, text=lang, variable=var)
            cb.grid(row=0, column=i, padx=2)
            self.checkboxes[lang] = var
        
        ttk.Label(main_frame, text="Pontuação Técnica (0-100):").grid(row=3, column=0, sticky=tk.W)
        self.score_tecnico = ttk.Scale(main_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.score_tecnico.grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Soft Skills (1-5):").grid(row=4, column=0, sticky=tk.W)
        self.softskills = ttk.Combobox(main_frame, values=list(range(1,6)))
        self.softskills.grid(row=4, column=1, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Meses Desempregado:").grid(row=5, column=0, sticky=tk.W)
        self.desemprego = ttk.Entry(main_frame)
        self.desemprego.grid(row=5, column=1, padx=5, pady=5)
        
        # Botões
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=15)
        ttk.Button(btn_frame, text="Classificar", command=self.classificar).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Limpar", command=self.limpar_formulario).pack(side=tk.LEFT)
        
        # Resultados
        self.resultado = scrolledtext.ScrolledText(main_frame, width=60, height=10, wrap=tk.WORD)
        self.resultado.grid(row=7, column=0, columnspan=2)
        
    def processar_entrada(self):
        try:
            entrada = {
                'experiencia_anos': int(self.experiencia.get()),
                'formacao': self.formacao.get(),
                'score_tecnico': round(float(self.score_tecnico.get())),
                'score_softskills': int(self.softskills.get()),
                'tempo_ultimo_emprego': int(self.desemprego.get()),
                'linguagens': [lang for lang, var in self.checkboxes.items() if var.get()]
            }
            
            df_entrada = pd.DataFrame([entrada])
            df_entrada['formacao'] = df_entrada['formacao'].map({'Graduação':0, 'Mestrado':1, 'Bootcamp':2})
            
            # Codificar linguagens corretamente
            linguagens_encoded = pd.DataFrame(
                mlb.transform([entrada['linguagens']]),  # Garantir lista de listas
                columns=mlb.classes_
            )
            
            df_final = pd.concat([df_entrada.drop(['linguagens'], axis=1), linguagens_encoded], axis=1)
            df_final = df_final.reindex(columns=X.columns, fill_value=0)
            
            return df_final
            
        except Exception as e:
            messagebox.showerror("Erro", f"Entrada inválida: {str(e)}")
            return None
    
    def classificar(self):
        self.resultado.delete(1.0, tk.END)
        dados = self.processar_entrada()
        
        if dados is not None:
            try:
                # Previsão
                predicao = model.predict(dados)[0]
                proba = model.predict_proba(dados)[0]
                
                # Explicação SHAP
                shap_valores = explainer.shap_values(dados)
                idx_classe = list(model.classes_).index(predicao)
                
                # Correção de extração de valores SHAP
                if isinstance(shap_valores, list):
                    valores = shap_valores[idx_classe][0]
                else:
                    valores = shap_valores[0, :, idx_classe]
                
                # Top 3 features
                feature_names = X.columns.tolist()
                top_features = np.argsort(-np.abs(valores))[:3]
                
                explicacoes = []
                for idx in top_features:
                    feature = feature_names[idx]
                    valor = dados.iloc[0][feature]
                    impacto = valores[idx]
                    
                    traducao = {
                        'formacao': 'Formação',
                        'score_tecnico': 'Pontuação Técnica',
                        'score_softskills': 'Soft Skills',
                        'tempo_ultimo_emprego': 'Meses Desempregado'
                    }.get(feature, feature)
                    
                    explicacoes.append(f"{traducao}: {valor} (Impacto: {impacto:.2f})")
                
                # Monta texto de saída
                resultado = f"""
                Resultado da Classificação:
                → Decisão: {predicao}
                → Probabilidades:
                - Aprovado: {proba[0]:.2%}
                - Aprovado Parcialmente: {proba[1]:.2%}
                - Reprovado: {proba[2]:.2%}
                
                Fatores Decisivos:
                """
                resultado += "\n".join([f"• {exp}" for exp in explicacoes])
                
                self.resultado.insert(tk.END, resultado)
                
            except Exception as e:
                self.resultado.insert(tk.END, f"Erro na classificação: {str(e)}")

    
    def limpar_formulario(self):
        self.experiencia.delete(0, tk.END)
        self.formacao.set('')
        self.score_tecnico.set(0)
        self.softskills.set('')
        self.desemprego.delete(0, tk.END)
        for var in self.checkboxes.values():
            var.set(False)
        self.resultado.delete(1.0, tk.END)

# Executar aplicativo
if __name__ == '__main__':
    root = tk.Tk()
    app = AplicativoRH(root)
    root.mainloop()