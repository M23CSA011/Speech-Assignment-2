# Speech-Assignment-2
## Question 1. Speaker Verification

### Goal
In speaker verification, the training dataset consists of audio clips paired with speaker IDs, denoted as (D = (xi, yi)). Given an audio clip (x) and a reference clip (x0), the objective is to ascertain whether (x0) and (x) belong to the same speaker.

### Tasks
- Choose three pre-trained models from the list: 'ecapa tdnn', 'hubert large', 'wav2vec2 xlsr', 'unispeech sat', 'wavlm base plus', 'wavlm large' trained on the VoxCeleb1 dataset.
- Calculate the EER(%) on the VoxCeleb1-H dataset using the above selected models.
- Compare your result with Table II of the WavLM paper.
- Evaluate the selected models on the test set of any one Indian language of the Kathbath Dataset. Report the EER(%).
- Fine-tune the best model on the validation set of the selected language of Kathbath Dataset. Report the EER(%).
- Provide an analysis of the results along with plausible reasons for the observed outcomes.

### Question 2. Source Separation

#### Goal
The goal of speech separation is to estimate individual speaker signals from their mixture, where the source signals may be overlapped with each other entirely or partially.

#### Tasks
- Generate the LibriMix dataset by combining two speakers from the LibriSpeech dataset, focusing solely on the LibriSpeech test clean partition.
- Partition the resulting LibriMix dataset into a 70-30 split for training and testing purposes. Evaluate the performance of the pre-trained SepFormer on the testing set, employing scale-invariant signal-to-noise ratio improvement (SISNRi) and signal-to-distortion ratio improvement (SDRi) as metrics.
- Fine-tune the SepFormer model using the training set and report its performance on the test split of the LibriMix dataset.
- Provide observations on the changes in performance throughout the experiment.
