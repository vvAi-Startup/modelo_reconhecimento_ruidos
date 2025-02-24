import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from datetime import datetime
import json
from scipy import signal
from tqdm import tqdm
import augment_audio as aug
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

def processar_arquivo_worker(args):
    """Função worker para processamento paralelo de arquivos"""
    arquivo, classe_idx, config = args
    try:
        # Carregar áudio
        y, sr = librosa.load(arquivo, sr=config['sr'], 
                           duration=config['duration'])
        
        # Verificar se o áudio foi carregado corretamente
        if y is None or len(y) == 0:
            print(f"Arquivo vazio ou inválido: {arquivo}")
            return None
        
        # Padding se necessário
        if len(y) < config['sr'] * config['duration']:
            y = np.pad(y, (0, config['sr'] * config['duration'] - len(y)))
        
        # Normalização
        y = librosa.util.normalize(y)
        
        # Gerar mel-spectrograma
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=config['sr'],
            n_mels=config['n_mels'],
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            fmin=config['fmin'],
            fmax=config['fmax']
        )
        
        # Converter para dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Redimensionar para tamanho fixo
        if mel_spec_db.shape != (config['n_mels'], config['n_mels']):
            mel_spec_db = librosa.util.fix_length(mel_spec_db, size=config['n_mels'], axis=1)
        
        return mel_spec_db, classe_idx
        
    except Exception as e:
        print(f"Erro ao processar {arquivo}: {e}")
        return None

class AudioClassifier:
    def __init__(self, config=None):
        self.config = {
            'sr': 16000,  # Taxa de amostragem
            'duration': 2,  # Duração em segundos
            'n_mels': 128,  # Número de bandas mel
            'n_fft': 1024,
            'hop_length': 128,
            'fmin': 80,    # Frequência mínima para voz
            'fmax': 8000,  # Frequência máxima para voz
            'batch_size': 32,
            'epochs': 1,
            'validation_split': 0.2,
            'test_split': 0.1,
            'model_path': 'modelos/',
            'log_path': 'logs/'
        } if config is None else config
        
        # Criar diretórios necessários
        os.makedirs(self.config['model_path'], exist_ok=True)
        os.makedirs(self.config['log_path'], exist_ok=True)
        
        # Inicializar métricas
        self.metrics_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def processar_audio(self, arquivo):
        """Processamento avançado de áudio"""
        try:
            # Carregar áudio
            y, sr = librosa.load(arquivo, sr=self.config['sr'], 
                               duration=self.config['duration'])
            
            # Padding se necessário
            if len(y) < self.config['sr'] * self.config['duration']:
                y = np.pad(y, (0, self.config['sr'] * self.config['duration'] - len(y)))
            
            # Remover silêncio
            y, _ = librosa.effects.trim(y, top_db=20)
            
            # Normalização
            y = librosa.util.normalize(y)
            
            # Filtro passa-banda para voz com verificação de frequências
            nyq = self.config['sr'] / 2
            low = self.config['fmin'] / nyq
            high = self.config['fmax'] / nyq
            
            # Garantir que as frequências estão no intervalo válido
            low = max(0.001, min(0.999, low))
            high = max(0.001, min(0.999, high))
            
            # Verificar se as frequências são válidas
            if low >= high:
                print(f"Aviso: Frequências inválidas para {arquivo}. Usando áudio sem filtro.")
            else:
                # Aplicar filtro apenas se as frequências forem válidas
                b, a = signal.butter(4, [low, high], btype='band')
                y = signal.filtfilt(b, a, y)
            
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
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
            
            # Normalização do espectrograma
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            
            return mel_spec_db
            
        except Exception as e:
            print(f"Erro no processamento do arquivo {arquivo}: {str(e)}")
            return None

    def processar_audio_paralelo(self, arquivo):
        """Processa um único arquivo de áudio de forma segura"""
        try:
            # Carregar áudio
            y, sr = librosa.load(arquivo, sr=self.config['sr'], 
                               duration=self.config['duration'])
            
            # Verificar se o áudio foi carregado corretamente
            if y is None or len(y) == 0:
                print(f"Arquivo vazio ou inválido: {arquivo}")
                return None
            
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
                mel_spec_db = librosa.util.fix_length(mel_spec_db, size=self.config['n_mels'], axis=1)
            
            return mel_spec_db
            
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {str(e)}")
            return None

    def aumentar_dados(self, audio, sr):
        """Técnicas de aumento de dados"""
        aumentados = []
        
        # Time stretching
        aumentados.append(librosa.effects.time_stretch(audio, rate=0.9))
        aumentados.append(librosa.effects.time_stretch(audio, rate=1.1))
        
        # Pitch shifting
        aumentados.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2))
        aumentados.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=2))
        
        # Adicionar ruído
        ruido = np.random.normal(0, 0.005, len(audio))
        aumentados.append(audio + ruido)
        
        # Time masking
        mask = aug.time_mask(audio)
        aumentados.append(mask)
        
        # Frequency masking
        mask = aug.freq_mask(audio)
        aumentados.append(mask)
        
        return aumentados

    def aumentar_dados_seguro(self, audio, sr):
        """Versão segura do aumento de dados"""
        try:
            aumentados = []
            
            # Time stretching com verificação
            try:
                audio_stretch = librosa.effects.time_stretch(audio, rate=0.9)
                if audio_stretch is not None and len(audio_stretch) > 0:
                    aumentados.append(audio_stretch)
            except Exception as e:
                print(f"Erro no time stretch: {e}")
            
            # Pitch shifting
            try:
                audio_pitch = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
                if audio_pitch is not None and len(audio_pitch) > 0:
                    aumentados.append(audio_pitch)
            except Exception as e:
                print(f"Erro no pitch shift: {e}")
            
            # Ruído
            try:
                audio_noise = audio + np.random.normal(0, 0.005, len(audio))
                aumentados.append(audio_noise)
            except Exception as e:
                print(f"Erro ao adicionar ruído: {e}")
            
            return aumentados
        except Exception as e:
            print(f"Erro no aumento de dados: {e}")
            return []

    def criar_modelo(self):
        """Criar modelo CNN avançado"""
        modelo = models.Sequential([
            # Primeira camada convolucional
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=(self.config['n_mels'], self.config['n_mels'], 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Segunda camada convolucional
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Terceira camada convolucional
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Camadas densas
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ])
        
        # Compilar modelo
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return modelo

    def carregar_dados(self, diretorio):
        """Carregar e preparar dados usando processamento paralelo"""
        dados = []
        rotulos = []
        classes = ['ruido', 'voz']
        
        # Preparar lista de arquivos e suas classes
        arquivos_e_classes = []
        for idx, classe in enumerate(classes):
            caminho = os.path.join(diretorio, classe)
            if not os.path.exists(caminho):
                raise ValueError(f"Diretório não encontrado: {caminho}")
            
            arquivos = [os.path.join(caminho, f) for f in os.listdir(caminho) 
                       if f.endswith(('.wav', '.mp3'))]
            # Adicionar config para cada arquivo
            arquivos_e_classes.extend([(f, idx, self.config) for f in arquivos])
        
        # Processar arquivos em paralelo
        n_cores = mp.cpu_count()
        print(f"Utilizando {n_cores} cores para processamento")
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            resultados = list(tqdm(
                executor.map(processar_arquivo_worker, arquivos_e_classes),
                total=len(arquivos_e_classes),
                desc="Processando arquivos"
            ))
        
        # Consolidar resultados válidos
        for resultado in resultados:
            if resultado is not None:
                spec, classe_idx = resultado
                if spec is not None:
                    dados.append(spec)
                    rotulos.append(classe_idx)
        
        if not dados:
            raise ValueError("Nenhum dado foi processado com sucesso")
        
        return np.array(dados), np.array(rotulos)

    def treinar(self, X, y):
        """Treinar modelo com validação cruzada e processamento paralelo"""
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_split'], random_state=42
        )
        
        # Expandir dimensões
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        
        # Calcular pesos das classes
        pesos_classes = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        pesos_dict = dict(enumerate(pesos_classes))
        
        # Data e hora para identificação única do modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(self.config['model_path'], 
                                    f'modelo_melhor_{timestamp}.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.config['model_path'], 
                                    f'ultimo_checkpoint_{timestamp}.keras'),
                save_freq='epoch',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Criar e treinar modelo
        modelo = self.criar_modelo()
        historico = modelo.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=self.config['validation_split'],
            callbacks=callbacks,
            class_weight=pesos_dict,
            verbose=1
        )
        
        # Avaliar modelo
        resultado = modelo.evaluate(X_test, y_test)
        predicoes = modelo.predict(X_test)
        predicoes_classes = np.argmax(predicoes, axis=1)
        
        # Gerar relatório
        report = classification_report(y_test, predicoes_classes)
        
        # Salvar o modelo final
        modelo_final_path = os.path.join(self.config['model_path'], 
                                        f'modelo_final_{timestamp}.keras')
        modelo.save(modelo_final_path)
        print(f"\nModelo final salvo em: {modelo_final_path}")
        
        # Salvar configuração
        config_path = os.path.join(self.config['model_path'], 
                                  f'config_{timestamp}.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Configuração salva em: {config_path}")
        
        # Salvar resultados
        self.salvar_resultados(historico, resultado, report)
        
        return modelo, historico

    def salvar_resultados(self, historico, resultado, report):
        """Salvar resultados do treinamento"""
        data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar métricas
        metricas = {
            'acuracia_teste': float(resultado[1]),
            'perda_teste': float(resultado[0]),
            'historico': {
                'acuracia': historico.history['accuracy'],
                'perda': historico.history['loss'],
                'val_accuracy': historico.history['val_accuracy'],
                'val_loss': historico.history['val_loss']
            },
            'report': report,
            'data_treinamento': data_hora
        }
        
        with open(os.path.join(self.config['log_path'], 
                              f'metricas_{data_hora}.json'), 'w') as f:
            json.dump(metricas, f, indent=4)

if __name__ == '__main__':
    # Configuração
    config = {
        'sr': 16000,
        'duration': 2,
        'n_mels': 128,
        'n_fft': 1024,
        'hop_length': 128,
        'fmin': 80,
        'fmax': 8000,
        'batch_size': 32,
        'epochs': 5,
        'validation_split': 0.2,
        'test_split': 0.1,
        'model_path': 'modelos/',
        'log_path': 'logs/'
    }
    
    # Inicializar classificador
    classificador = AudioClassifier(config)
    
    # Carregar e preparar dados
    X, y = classificador.carregar_dados('./dataset/')
    
    # Treinar modelo
    modelo, historico = classificador.treinar(X, y) 