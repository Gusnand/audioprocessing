import librosa  # Library untuk pemrosesan audio
import glob  # Library untuk pencarian file berdasarkan pola
import numpy as np  # Library untuk manipulasi array

#Keras--> High-level neural networks API, yg berjalan di atas framework deep learning seperti TensorFLow.
#Library Keras ini memberikan kemudahan dalam build dan training neural network/JST
from keras.models import Sequential  # Modul Keras untuk model sequential
from keras.layers import Dense  # Modul Keras untuk layer Dense
from keras.optimizers import SGD  # Modul Keras untuk optimizer SGD
from keras.utils import to_categorical  # Modul Keras untuk one-hot encoding

from sklearn.model_selection import train_test_split  # Modul scikit-learn untuk pembagian data train dan test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Modul scikit-learn untuk evaluasi model

import streamlit as st  # Library untuk membuat aplikasi web interaktif


def create_model(input_shape, num_classes, num_hidden_layers, num_hidden_neurons, activation):
    """
    Membuat model neural network dengan jumlah hidden layer, jumlah neuron, dan fungsi aktivasi yang ditentukan.
    """
    model = Sequential()
    model.add(Dense(num_hidden_neurons, activation=activation, input_shape=input_shape))
    
    for _ in range(num_hidden_layers - 1):
        model.add(Dense(num_hidden_neurons, activation=activation))
    
    model.add(Dense(num_classes, activation='softmax'))
    return model


def train_model(model, X_train, y_train, lr, num_epochs):
    """
    Melatih model dengan data train menggunakan learning rate dan jumlah epochs yang ditentukan.
    """
    optimizer = SGD(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, verbose=0)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Mengevaluasi model menggunakan data test dan menghitung metrik evaluasi seperti akurasi, presisi, recall, dan F1-Score.
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1


def preprocess_audio(audio_path, label):
    """
    Melakukan preprocessing audio dengan menghitung fitur MFCC dari file audio.
    """
    signal, sample_rate = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean, label


def load_data(path):
    """
    Memuat data audio dari file .wav sesuai dengan path yang diberikan.
    """
    audio_files = glob.glob(path)
    X = []
    y = []
    
    for audio_file in audio_files:
        label = 0 if 'happy' in audio_file else 1
        mfccs_mean, label = preprocess_audio(audio_file, label)
        X.append(mfccs_mean)
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def deploy_web_app(best_model):
    """
    Mendeploy model terbaik ke dalam aplikasi web menggunakan Streamlit.
    """
    st.title("Identifikasi Sentimen dari Suara")
    st.write("Aplikasi ini menggunakan model neural network untuk mengidentifikasi sentimen atau emosi dari suara.")
    
    audio_files = st.file_uploader("Upload satu atau beberapa file audio", type=".wav", accept_multiple_files=True)
    
    if audio_files is not None:
        for audio_file in audio_files:
            signal, sample_rate = librosa.load(audio_file)
            mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            input_data = np.array([mfccs_mean])
            sentiment = "Positive" if np.argmax(best_model.predict(input_data)) == 0 else "Negative"
            
            st.write("Hasil identifikasi sentimen:")
            st.write(f"File: {audio_file.name}")
            st.write(f"Sentimen: {sentiment}")
            st.write("=========================================")


def main(path):
    """
    Fungsi utama yang menjalankan alur aplikasi.
    """
    X, y = load_data(path)
    
    num_classes = 2  # Jumlah kelas (positive sentiment dan negative sentiment)
    input_shape = (13,)  # Bentuk data input (fitur MFCC)

    num_samples = len(X)
    num_test_samples = int(0.2*num_samples)

    if num_samples > 0 and num_test_samples > 0:
        test_size = num_test_samples / num_samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
    
    else:
        print("Error: Data tidak cukup untuk dibagi menjadi data training dan data testing.")
    
    learning_rates = [0.01, 0.001, 0.0001]
    num_epochs_list = [50, 100, 200]
    hidden_layers_list = [1, 2, 3]
    hidden_neurons_list = [32, 64, 128]
    activations = ['sigmoid', 'softmax', 'tanh']
    
    best_accuracy = 0
    best_model = None
    
    for lr in learning_rates:
        for num_epochs in num_epochs_list:
            for num_hidden_layers in hidden_layers_list:
                for num_hidden_neurons in hidden_neurons_list:
                    for activation in activations:
                        model = create_model(input_shape, num_classes, num_hidden_layers, num_hidden_neurons, activation)
                        model = train_model(model, X_train, y_train, lr, num_epochs)
                        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model
                        
                        print(f"Learning Rate: {lr}, Epochs: {num_epochs}, Hidden Layers: {num_hidden_layers}, Hidden Neurons: {num_hidden_neurons}, Activation: {activation}")
                        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
                        print("=========================================")
    
    deploy_web_app(best_model)


# Contoh penggunaan
main('FP-AudioJST/audiosamples/*.wav')
