import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import sounddevice as sd
from scipy.io.wavfile import write
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import json
from datetime import datetime

class TestadorModelo:
    def __init__(self):
        self.modelo = None
        self.config = None
        self.carregar_configuracao_mais_recente()
        self.carregar_modelo_mais_recente()
        
    def carregar_configuracao_mais_recente(self):
        """Carrega a configuração mais recente do modelo"""
        try:
            pasta_modelos = 'modelos'
            
            # Verificar se a pasta existe
            if not os.path.exists(pasta_modelos):
                print(f"Pasta '{pasta_modelos}' não encontrada. Criando...")
                os.makedirs(pasta_modelos)
                raise FileNotFoundError("Pasta de modelos criada, mas não há configurações")
            
            # Listar todos os arquivos de configuração
            arquivos_config = [f for f in os.listdir(pasta_modelos) 
                             if f.startswith('config_') and f.endswith('.json')]
            
            print(f"Arquivos de configuração encontrados: {arquivos_config}")
            
            if not arquivos_config:
                raise FileNotFoundError("Nenhuma configuração encontrada")
            
            # Pega o arquivo mais recente
            config_mais_recente = max(arquivos_config)
            caminho_config = os.path.join(pasta_modelos, config_mais_recente)
            
            with open(caminho_config, 'r') as f:
                self.config = json.load(f)
            print(f"Configuração carregada: {caminho_config}")
            
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
            print("Usando configuração padrão...")
            # Configuração padrão
            self.config = {
                'sr': 16000,
                'duration': 2,
                'n_mels': 128,
                'n_fft': 1024,
                'hop_length': 128,
                'fmin': 20,
                'fmax': 7000
            }
    
    def carregar_modelo_mais_recente(self):
        """Carrega o modelo mais recente da pasta modelos"""
        try:
            pasta_modelos = 'modelos'
            
            # Verificar se a pasta existe
            if not os.path.exists(pasta_modelos):
                print(f"Pasta '{pasta_modelos}' não encontrada")
                return False
            
            # Listar todos os arquivos de modelo
            arquivos_modelo = [f for f in os.listdir(pasta_modelos) 
                             if (f.startswith('modelo_melhor_') or f.startswith('modelo_final_')) 
                             and f.endswith('.keras')]
            
            print(f"Arquivos de modelo encontrados: {arquivos_modelo}")
            
            if not arquivos_modelo:
                print("Nenhum modelo encontrado na pasta")
                return False
            
            # Pega o arquivo mais recente
            modelo_mais_recente = max(arquivos_modelo)
            caminho_modelo = os.path.join(pasta_modelos, modelo_mais_recente)
            
            print(f"Tentando carregar modelo: {caminho_modelo}")
            self.modelo = load_model(caminho_modelo)
            print(f"Modelo carregado com sucesso: {caminho_modelo}")
            return True
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            print("Detalhes do erro:", str(e))
            return False
    
    def processar_audio(self, arquivo):
        """Processa o arquivo de áudio para classificação"""
        try:
            # Carregar áudio
            y, sr = librosa.load(arquivo, sr=self.config['sr'], 
                               duration=self.config['duration'])
            
            # Padding se necessário
            if len(y) < self.config['sr'] * self.config['duration']:
                y = np.pad(y, (0, self.config['sr'] * self.config['duration'] - len(y)))
            
            # Normalização
            y = librosa.util.normalize(y)
            
            # Gerar mel-spectrograma
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.config['sr'],
                n_mels=self.config['n_mels'],
                n_fft=self.config['n_fft'],
                hop_length=self.config['hop_length'],
                fmin=self.config['fmin'],
                fmax=self.config['fmax']
            )
            
            # Converter para dB
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Redimensionar para tamanho fixo
            if mel_spec_db.shape != (self.config['n_mels'], self.config['n_mels']):
                mel_spec_db = librosa.util.fix_length(mel_spec_db, 
                                                    size=self.config['n_mels'], 
                                                    axis=1)
            
            # Expandir dimensões para o modelo
            mel_spec_db = mel_spec_db[..., np.newaxis]
            
            return mel_spec_db
            
        except Exception as e:
            print(f"Erro no processamento do áudio: {e}")
            return None
    
    def classificar(self, audio_path):
        """Classifica um arquivo de áudio"""
        if self.modelo is None:
            return "Erro: Modelo não carregado"
        
        try:
            # Processa o áudio
            features = self.processar_audio(audio_path)
            if features is None:
                return "Erro no processamento do áudio"
            
            # Adiciona dimensão do batch
            features = features[np.newaxis, ...]
            
            # Realiza múltiplas predições com pequenas variações
            num_predicoes = 5
            predicoes = []
            
            # Predição original
            predicoes.append(self.modelo.predict(features, verbose=0))
            
            # Predições com pequenas variações de ruído
            for _ in range(num_predicoes - 1):
                features_noise = features + np.random.normal(0, 0.01, features.shape)
                predicoes.append(self.modelo.predict(features_noise, verbose=0))
            
            # Média das predições
            predicao_media = np.mean(predicoes, axis=0)
            
            classes = ['ruido', 'voz']
            prob_voz = predicao_media[0][1]
            prob_ruido = predicao_media[0][0]
            
            # Sistema de votação com histerese
            if prob_voz > 0.6:  # Confiança alta para voz
                classe_predita = 'voz'
            elif prob_voz < 0.3:  # Confiança alta para ruído
                classe_predita = 'ruido'
            else:  # Zona de incerteza
                classe_predita = 'voz' if prob_voz > 0.45 else 'ruido'
            
            # Formatação do resultado
            resultado = f"Classificação: {classe_predita}\n"
            resultado += f"Probabilidade Voz: {prob_voz*100:.2f}%\n"
            resultado += f"Probabilidade Ruído: {prob_ruido*100:.2f}%\n"
            
            # Adiciona indicador de confiança
            confianca = max(prob_voz, prob_ruido)
            if confianca > 0.8:
                resultado += "Confiança: Alta\n"
            elif confianca > 0.6:
                resultado += "Confiança: Média\n"
            else:
                resultado += "Confiança: Baixa\n"
            
            return resultado
            
        except Exception as e:
            return f"Erro na classificação: {e}"

class Interface:
    def __init__(self):
        self.testador = TestadorModelo()
        self.gravando = False
        self.criar_interface()
    
    def criar_interface(self):
        """Cria a interface gráfica"""
        self.janela = tk.Tk()
        self.janela.title("Testador de Classificação Voz/Ruído")
        self.janela.geometry("400x500")
        
        # Estilo
        style = ttk.Style()
        style.configure('Custom.TButton', padding=10)
        
        # Frame principal
        frame = ttk.Frame(self.janela, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Botões
        ttk.Button(frame, 
                  text="Selecionar Arquivo", 
                  command=self.selecionar_arquivo,
                  style='Custom.TButton').pack(fill=tk.X, pady=5)
        
        self.btn_gravar = ttk.Button(frame, 
                                   text="Iniciar Gravação", 
                                   command=self.toggle_gravacao,
                                   style='Custom.TButton')
        self.btn_gravar.pack(fill=tk.X, pady=5)
        
        # Área de resultado
        ttk.Label(frame, text="Resultado:").pack(pady=5)
        self.resultado_text = tk.Text(frame, height=10, wrap=tk.WORD)
        self.resultado_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Label de status
        self.status_label = ttk.Label(frame, text="Pronto")
        self.status_label.pack(pady=10)
        
        self.janela.mainloop()
    
    def selecionar_arquivo(self):
        """Permite selecionar um arquivo de áudio"""
        arquivo = filedialog.askopenfilename(
            filetypes=[("Arquivos de áudio", "*.wav *.mp3")])
        if arquivo:
            self.atualizar_status("Processando arquivo...")
            resultado = self.testador.classificar(arquivo)
            self.mostrar_resultado(resultado)
            self.atualizar_status("Pronto")
    
    def toggle_gravacao(self):
        """Inicia/para a gravação de áudio"""
        if not self.gravando:
            self.gravando = True
            self.btn_gravar.configure(text="Parar Gravação")
            self.atualizar_status("Gravando...")
            self.gravar_audio()
        else:
            self.gravando = False
            self.btn_gravar.configure(text="Iniciar Gravação")
            self.atualizar_status("Processando gravação...")
    
    def gravar_audio(self):
        """Grava áudio do microfone"""
        try:
            # Configurações de gravação
            duracao = self.testador.config['duration']
            sr = self.testador.config['sr']
            
            # Gravar áudio
            gravacao = sd.rec(int(sr * duracao), samplerate=sr, channels=1)
            sd.wait()
            
            # Salvar temporariamente
            nome_arquivo = f"temp_gravacao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            write(nome_arquivo, sr, gravacao)
            
            # Classificar
            resultado = self.testador.classificar(nome_arquivo)
            self.mostrar_resultado(resultado)
            
            # Limpar arquivo temporário
            os.remove(nome_arquivo)
            
            self.atualizar_status("Pronto")
            self.gravando = False
            self.btn_gravar.configure(text="Iniciar Gravação")
            
        except Exception as e:
            self.mostrar_resultado(f"Erro na gravação: {e}")
            self.atualizar_status("Erro")
            self.gravando = False
            self.btn_gravar.configure(text="Iniciar Gravação")
    
    def mostrar_resultado(self, texto):
        """Atualiza a área de resultado"""
        self.resultado_text.delete(1.0, tk.END)
        self.resultado_text.insert(tk.END, texto)
    
    def atualizar_status(self, texto):
        """Atualiza o label de status"""
        self.status_label.config(text=texto)

if __name__ == "__main__":
    interface = Interface() 