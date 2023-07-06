import matplotlib.pyplot as plt
import librosa.display
import os
import librosa
import soundfile as sf
import glob
import numpy as np
import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.models import save_model, load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st

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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_test = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1

def preprocess_audio(audio_path, label, target_sample_rate):
    """
    Melakukan preprocessing audio dengan menghitung fitur MFCC dari file audio.
    Mengubah sampling rate audio menjadi target_sample_rate.
    """
    signal, sample_rate = sf.read(audio_path)
    signal_resampled = librosa.resample(y=signal, orig_sr=sample_rate, target_sr=target_sample_rate)
    signal_normalized = librosa.util.normalize(signal_resampled)
    mfccs = librosa.feature.mfcc(y=signal_normalized, sr=target_sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean, label

def load_data(path, target_sample_rate):
    """
    Memuat data audio dari file .wav sesuai dengan path yang diberikan.
    """
    audio_files = glob.glob(path)
    X = []
    y = []

    for audio_file in audio_files:
        signal, sample_rate = sf.read(audio_file)
        signal_resampled = librosa.resample(y=signal, orig_sr=sample_rate, target_sr=target_sample_rate)
        signal_normalized = librosa.util.normalize(signal_resampled)
        mfccs = librosa.feature.mfcc(y=signal_normalized, sr=target_sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        X.append(mfccs_mean)
        y.append(0 if 'happy' in audio_file else 1)

    X = np.array(X)
    y = np.array(y)

    return X, y



def deploy_web_app(best_model, X_test, y_test, target_sample_rate):
    """
    Mendeploy model terbaik ke dalam aplikasi web menggunakan Streamlit.
    """
    st.title("Identifikasi Sentimen dari Suara")
    st.write("Aplikasi ini menggunakan model neural network untuk mengidentifikasi sentimen atau emosi dari suara.")

    audio_files = st.file_uploader("Upload satu atau beberapa file audio", type=".wav", accept_multiple_files=True)

    if audio_files is not None:
        for audio_file in audio_files:
            signal, sample_rate = sf.read(audio_file)
            signal_resampled = librosa.resample(y=signal, orig_sr=sample_rate, target_sr=target_sample_rate)
            signal_normalized = librosa.util.normalize(signal_resampled)
            mfccs = librosa.feature.mfcc(y=signal_normalized, sr=target_sample_rate, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            input_data = np.array([mfccs_mean])
            sentiment = "Positive" if np.argmax(best_model.predict(input_data)) == 0 else "Negative"

            accuracy, precision, recall, f1 = evaluate_model(best_model, X_test, y_test)
            st.write("Hasil identifikasi sentimen:")
            st.write(f"File: {audio_file.name}")
            st.write(f"Sentimen: {sentiment}")
            st.write(f"Akurasi: {accuracy}")
            st.write(f"Presisi: {precision}")
            st.write(f"Recall: {recall}")
            st.write(f"F1-Score: {f1}")
            st.write("=========================================")

            st.write("Cepstral Coefficients:")
            for i, coef in enumerate(mfccs_mean):
                st.write(f"MFCC {i+1}: {coef}")

            st.write("MFCC Chart Analysis:")
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfccs, x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('MFCC')
            plt.xlabel('Time')
            plt.ylabel('MFCC Coefficients')
            st.pyplot(plt)

            st.write("======================================================================================")

def main(path, target_sample_rate):
    """
    Fungsi utama yang menjalankan alur aplikasi.
    """
    project_directory = r"C:\Master D\Semester 4 (Killing Machine)\Pengantar Pemrosesan Data Multimedia\FP-AudioJST"
    audiosamples_directory = os.path.join(project_directory, "audiosamples")

    X, y = load_data(os.path.join(audiosamples_directory, "*.wav"), target_sample_rate)

    num_classes = 2  # Jumlah kelas (positive sentiment dan negative sentiment)
    input_shape = (13,)  # Bentuk data input (fitur MFCC)

    num_samples = len(X)
    num_test_samples = int(0.2*num_samples)

    if num_samples > 0 and num_test_samples > 0:
        test_size = num_test_samples / num_samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        learning_rates = [0.01, 0.001, 0.0001]
        num_epochs_list = [50, 100, 200]
        hidden_layers_list = [1, 2, 3]
        hidden_neurons_list = [32, 64, 128]
        activations = ['sigmoid', 'softmax', 'tanh']

        if os.path.exists('best_model.h5'):
            # Load the saved model
            best_model = load_model('best_model.h5')
            deploy_web_app(best_model, X_test, y_test, target_sample_rate)
        
        else:
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
        
            if best_model is not None:
                # Save the best model to a file
                save_model(best_model, 'best_model.h5')

                # Load the saved model
                loaded_model = load_model('best_model.h5')

                deploy_web_app(loaded_model, X_test, y_test, target_sample_rate)
            else:
                print("Error: Model terbaik tidak ditemukan.")

    else:
        print("Error: Data tidak cukup untuk dibagi menjadi data training dan data testing.")

# Contoh penggunaan
main('audiosamples/*.wav', 44100)  # Set the target sampling rate to 44100 Hz
