import pandas as pd
import numpy as np
import torch
import os
import librosa
import soundfile as sf
from transformers import AutoProcessor, ASTModel, AutoFeatureExtractor

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# #print(torch.cuda.get_device_name(0))

# if device.type == "cuda":
#     print(torch.cuda.get_device_name(device))

# use the cpu
device = torch.device("cpu")
print(f"Using device: {device}")

# load the pretrained ast model

#model_dir = "AST/model"
model_dir = r"C:\Users\julia\Documents\Deep Learning Project\Music-genre-classification-main\AST\model"
print("Model folder exists:", os.path.isdir(model_dir))

if not os.path.isdir(model_dir):
    raise FileNotFoundError(
        f"Model folder not found: {model_dir}\n"
        "Check where you saved your downloaded model files."
    )

print("Model files:", os.listdir(model_dir))

processor = AutoProcessor.from_pretrained(model_dir)
model = ASTModel.from_pretrained(model_dir)
#model = model.to("cuda")
model = model.to(device)
print(model.eval())
#------------------------------------------------------------------------
# find audio files
#path2 = 'Wav2vec2/GTzan_16k_Wav'
data_dir = r'C:\Users\julia\Documents\Deep Learning Project\Data\genres_original'
if not os.path.isdir(data_dir):
    raise FileNotFoundError(
        f"Dataset folder not found: {data_dir}"
    )

# Read a directory and put all files in a list
file_list = []
i = 0

for path, subdirs, files in os.walk(data_dir):
    for name in files:
        file_list.append(os.path.join(path, name))
        i += 1

# Skip the one unreadable audio file
file_list = [
    file for file in file_list
    if os.path.basename(file) != "jazz.00054.wav"
]
print("Files processed: "+str(i) )


#-------------------------------------------------------------------------
# sample rate = 16,000
#  1s = 16,000 x 1 =  16,000
# 60s = 16,000 x 60 = 960,500

length = list()
i      = 0
avg    = 0

for file in file_list:
    #data, samplerate = sf.read( file )
    # SOME FILES WERE UNREADABLE
    try:
        data, samplerate = sf.read(file)
    except Exception as e:
        print(f"Skipping unreadable file: {file}")
        print(e)
        continue
    #if len(data) <= 960500:                         
           # print("Audio length: "+str(len(data))+" with less than 30s: "+str(file) )
    #computer average lenght of files
    avg = avg + len(data)
    length.append(len(data))
    i += 1

print( "Files processed: "+str(i) )
print( "Average file length: "+str(avg/i) + " samples   "+str(avg/i/samplerate)+" s   "+str(avg/i/samplerate/60)+" min" )
print( "Max length: "+str(max(length))+ " samples   "+str(max(length)/samplerate)+" s   "+str(max(length)/samplerate/60)+" min" )
print( "Min length: "+str(min(length))+ " samples   "+str(min(length)/samplerate)+" s   "+str(min(length)/samplerate/60)+" min" )

#---------------------------------------------------------------------------
sampling_rate = 16000
track_count = 0
print('\n extraction started')

for file in file_list:

    filename = os.path.splitext(os.path.basename(file))[0]
    file_id = 'AST_3rdLayer/' + filename + '.csv'

    if os.path.exists(file_id):
        print("Already extracted, skipping:", filename)
        continue

    data, samplerate = sf.read(file)
    # Resample audio to 16000 Hz for AST
    if samplerate != 16000:
        data = librosa.resample(
            data,
            orig_sr=samplerate,
            target_sr=16000
        )
        samplerate = 16000
    print ("--------------")
    print ("Sample Rate: " + str(samplerate) + " Length: " + str(data.shape) + " " + str( file ) )

    input_values = processor(data, return_tensors="pt", sampling_rate=samplerate).input_values  
    #input_values = input_values.to("cuda") 
    input_values = input_values.to(device)
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states  
    
    rep6 = hidden_states[2]  # 3 layer
    rep6 = rep6.squeeze().cpu().numpy()
    rep_df = pd.DataFrame(rep6)
    filename = os.path.splitext(os.path.basename(file))[0]
    file_id = 'AST_3rdLayer/' + filename + '.csv'

    #file_id = 'AST/AST_3rdLayer/'+file.split('/')[2].split('.')[0]+'.'+file.split('/')[2].split('.')[1]+'.csv'
    rep_df.to_csv(file_id)
    print(file_id)
# #---------------------------------------
#     rep12 = hidden_states[5]  # 6 layer
#     rep12 = rep12.squeeze().cpu().numpy()
#     rep_df = pd.DataFrame(rep12)
#     #file_id = 'AST/AST_6thLayer/'+file.split('/')[2].split('.')[0]+'.'+file.split('/')[2].split('.')[1]+'.csv'
#     file_id = 'AST_6thLayer/' + filename + '.csv'
#     rep_df.to_csv(file_id)
#     print(file_id)
# #---------------------------------------
#     rep18 = hidden_states[8]  # 9 layer
#     rep18 = rep18.squeeze().cpu().numpy()
#     rep_df = pd.DataFrame(rep18)
#     #file_id = 'AST/AST_9thLayer/'+file.split('/')[2].split('.')[0]+'.'+file.split('/')[2].split('.')[1]+'.csv'
#     file_id = 'AST_9thLayer/' + filename + '.csv'
#     rep_df.to_csv(file_id)
#     #print(file_id)

    print( file_id, " ", str(track_count) )
    track_count += 1

# read wav audio tracks, pretrained ast model, extract features from 3rd, 6th, 9th layers, features saved in csv (will go into genre classifier)