# VAD
Voice Activity Detection


└── VAD
    ├── LICENSE
    ├── data_fbank
    │   ├── copy_noise.py
    │   ├── divide.py
    │   └── do_cmvn_and_delta.sh
    ├── log
    │   └── dnn256_2000h_noise_and_reverb_epoch4_test0.975.log
    ├── model
    │   ├── dnn256_2000h_noise_and_reverb_epoch4_test0.975.pkl
    │   └── model_old.pkl
    ├── src
    │   ├── ali_process.py
    │   ├── data_process.py
    │   ├── data_transform.py
    │   ├── do.sh
    │   ├── gpu06.log
    │   ├── main.py
    │   ├── network.py
    │   ├── stat.py
    │   └── util.py
    └── toJson
        ├── jsonToPytorchModel.py
        ├── pytorchModelToJson.py
        ├── vad128…_epoch78_0.928171.json
        └── vad_dnn_2000h_epoch3_0.975.json
