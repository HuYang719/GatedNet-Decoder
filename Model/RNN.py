import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Lambda
from keras.layers import SimpleRNN,LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from scipy.stats import levy_stable
from keras.optimizers import Adam
from Model import CommFunc
from Datasets import Data
from time import time
from Metrics import metrics



class RNNDecoder():
    def __init__(self, alpha=1.8, GSNR=1, RNNtype='LSTM'):
        self.k = 8
        self.N = 16
        self.R = self.k/self.N
        self.optimizer = Adam(0.0002, 0.5)
        self.loss = 'binary_crossentropy'
        self.GSNR_train = GSNR
        self.alpha_train = alpha
        self.scale_train=CommFunc.CalScale(self.GSNR_train, self.alpha_train, self.R)
        self.model, self.modulator_layers, self.noise, self.decoder_layers, self.decoder = self.build_Decoder(RNNtype)

    def return_output_shape(self, input_shape):
        return input_shape

    def compose_model(self, layers):
        model = Sequential()
        for layer in layers:
            model.add(layer)
        return model

    def build_Decoder(self,RNNtype):
        # Define modulator
        modulator_layers = [Lambda(CommFunc.modulateBPSK,
                                   input_shape=(None,1), output_shape=self.return_output_shape, name="modulator")]
        modulator = self.compose_model(modulator_layers)
        modulator.compile(optimizer=self.optimizer, loss=self.loss)

        # Define noise
        noise_layers = [Lambda(CommFunc.addNoise, arguments={'sigma': self.scale_train,'alpha_train':self.alpha_train},
                               input_shape=(None, 1), output_shape=self.return_output_shape, name="noise")]
        noise = self.compose_model(noise_layers)
        noise.compile(optimizer=self.optimizer, loss=self.loss)

        # Define decoder
        if RNNtype == 'LSTM':
            decoder_layers = [Bidirectional(LSTM(100, return_sequences=True, input_shape=(None, 1), recurrent_dropout=0.5))]
            # decoder_layers.append(BatchNormalization())
            decoder_layers.append(Bidirectional(LSTM(100, recurrent_dropout=0.5)))
            # decoder_layers.append(BatchNormalization())
        elif RNNtype == 'GRU':
            decoder_layers = [Bidirectional(GRU(200, return_sequences=True, input_shape=(None, 1), recurrent_dropout=0.5))]
            decoder_layers.append(Bidirectional(GRU(200, recurrent_dropout=0.5)))
        elif RNNtype == 'SimpleRNN':
            decoder_layers = [Bidirectional(SimpleRNN(200, return_sequences=True, input_shape=(None, 1), recurrent_dropout=0.5))]
            decoder_layers.append(Bidirectional(SimpleRNN(200, recurrent_dropout=0.5)))
        else:
            print("Wrong RNNtype! Only for LSTM/GRU/SimpleRNN !")
        decoder_layers.append(Dense(8))
        decoder = self.compose_model(decoder_layers)
        decoder.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER, metrics.errors])

        # Define model
        model_layers = modulator_layers + noise_layers + decoder_layers
        model = self.compose_model(model_layers)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER])
        model.summary()
        return model, modulator_layers, noise, decoder_layers, decoder

    def train(self, epochs=2**16, batch_size=256, GSNR=1,verbose=1):
        scale_train = CommFunc.CalScale(GSNR, self.alpha_train, self.R)

        noise_layers = [
            Lambda(CommFunc.addNoise, arguments={'sigma': scale_train, 'alpha_train': self.alpha_train},
                   input_shape=(None, 1), output_shape=self.return_output_shape, name="noise")]
        noise = self.compose_model(noise_layers)
        noise.compile(optimizer=self.optimizer, loss=self.loss)

        model_layers = self.modulator_layers + noise_layers + self.decoder_layers
        model = self.compose_model(model_layers)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER])


        X_train,Y_train = Data.genData(self.k,self.N,batch_size)
        X_train = X_train.reshape(-1,16,1)
        print(X_train.shape,Y_train.shape)
        X_val, Y_val = Data.genData(self.k, self.N, batch_size)
        X_val=X_val.reshape(-1,16,1)
        t = time()
        history = model.fit(X_train, Y_train,
                                 validation_data=[X_val,Y_val],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=verbose,
                                 shuffle=True)
        t=time()-t
        print("Time Used:{}s = {}min".format(t, t / 60))
        return self.model,history

    def test(self,alpha,GSNR_low,GSNR_up,interval,test_batch,num_words):
        np.seterr(divide='ignore', invalid='ignore')
        SNR_dB_start_Eb = GSNR_low
        SNR_dB_stop_Eb = GSNR_up
        SNR_points = interval
        SNR_dB_start_Es = SNR_dB_start_Eb + 10 * np.log10(self.k / self.N)
        SNR_dB_stop_Es = SNR_dB_stop_Eb + 10 * np.log10(self.k / self.N)
        SNRs = np.linspace(SNR_dB_start_Eb, SNR_dB_stop_Eb, SNR_points)

        sigma_start = np.sqrt(1 / (2 * 10 ** (SNR_dB_start_Es / 10)))
        sigma_stop = np.sqrt(1 / (2 * 10 ** (SNR_dB_stop_Es / 10)))

        sigmas = np.linspace(sigma_start, sigma_stop, SNR_points)

        nb_errors = np.zeros(len(sigmas), dtype=int)
        nb_bits = np.zeros(len(sigmas), dtype=int)
        ber = np.zeros(len(sigmas), dtype=float)
        seedrand = np.zeros(100, dtype=int)

        for sr in range(1, 100):
            seedrand[sr] = np.random.randint(0, 2 ** 14, size=(1))  # seedrand[sr-1]+1
        for i in range(0, len(sigmas)):  # different  SNR
            scale = CommFunc.CalScale(SNRs[i], alpha, self.R)
            # print("GSNR={},scale={}".format(SNRs[i], scale))
            for ii in range(0, np.round(num_words / test_batch).astype(int)):
                # Source
                x_test, d_test=Data.genRanData(self.k, self.N, test_batch, seedrand[ii])
                # Modulator (BPSK)
                s_test = -2 * x_test + 1
                # Channel (alpha-stable)
                y_test = s_test + levy_stable.rvs(alpha, 0, 0, scale, (test_batch, self.N))
                y_test = y_test.reshape(-1,16,1)
                # Decoder
                nb_errors[i] += self.decoder.evaluate(y_test, d_test, batch_size=test_batch, verbose=2)[2]
                nb_bits[i] += d_test.size
                ber = np.float32(nb_errors/nb_bits)
            # print(y_test[0:5],d_test[0:5])
        return ber,nb_bits,nb_errors


#
if __name__ == '__main__':
    LSTMDecoder=RNNDecoder(1.2, 1,'LSTM')
    LSTMDecoder.train(2**10, 256, 1,0)
    ber1,nb_bits,nb_errors = LSTMDecoder.test(1.2, 0, 5, 6, 100, 10000)
    print(ber1,nb_bits,nb_errors)
