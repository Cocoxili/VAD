# VAD based on DNN


Voice activity detection (VAD), also known as speech activity detection or speech detection, is a technique used in speech processing in which the presence or absence of human speech is detected. The main uses of VAD are in speech coding and speech recognition. It can facilitate speech processing, and can also be used to deactivate some processes during non-speech section of an audio session: it can avoid unnecessary coding/transmission of silence packets in Voice over Internet Protocol applications, saving on computation and on network bandwidth.

Voice activity detection (VAD) was first investigated for use on time-assignment speech interpolation (TASI) systems. VAD is an important step in speech processing applications such as mobile telephony, speech enhancement as well as speech and speaker recognition, and it is particularly important in the field of speech recognition.

Typical models for VAD include support vector machine (SVM), Gaussian mixture model (GMM), recurrent neural network (RNN) and deep neural network (DNN).

**This work is VAD based on DNN model which is implemented by pyTorch.**



### file structure

    ./
    ├── LICENSE
    ├── README.md
    ├── data_fbank
    │   ├── copy_noise.py
    │   ├── divide.py
    │   └── do_cmvn_and_delta.sh
    ├── log
    │   ├── Acc.png
    │   ├── dnn256_2000h_noise_and_reverb_epoch4_test0.975.log
    │   ├── testACC.log
    │   └── trainACC.log
    ├── model
    │   ├── dnn_2000h_v4.pkl
    │   └── model_old.pkl
    ├── src
    │   ├── ali_process.py
    │   ├── data_process.py
    │   ├── data_transform.py
    │   ├── do.sh
    │   ├── main.py
    │   ├── network.py
    │   ├── stat.py
    │   └── util.py
    └── toJson
        ├── jsonToPytorchModel.py
        ├── pytorchModelToJson.py
        ├── vad128_128_SNR98_db0_25_RESCALE-45_-15_V1_20170825201833_epoch78_0.928171.json
        └── vad_dnn256_2000h_v2.json

        
 
**Train and Test accuracy**

![TrainTestAcc](https://github.com/Black-Black-Man/VAD/blob/master/log/Acc.png?raw=true)


